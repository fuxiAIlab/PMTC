import os
import sys

sys.path.append('..')
import ipdb
import collections

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from packaging import version
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm, trange

from transformers import Trainer
from transformers.utils import logging

from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    HPSearchBackend,
    TrainOutput,
    set_seed,
)

from .data_collator import DataCollatorForGameMultiTask
from app_utils import get_target_loss

logger = logging.get_logger(__name__)

_use_native_amp = False
_use_apex = False


class TasksLossTracker():
    def __init__(self, save_path=''):
        self.step_loss = collections.defaultdict(lambda: [])
        self.batch_loss = collections.defaultdict(lambda: [])
        self.save_path = save_path

    def add_batch_loss(self, task_name, value):
        self.batch_loss[task_name].append(value)

    def compute_batch_loss(self):
        for task_name, loss_values in self.batch_loss.items():
            loss_values = [float(x) for x in loss_values]
            avg_step_loss = np.average(loss_values)
            self.step_loss[task_name].append(avg_step_loss)
        self.batch_loss = collections.defaultdict(lambda: [])

    def add_curr_info(self, curr_learner):
        self.step_loss['curr_task_info'].append(curr_learner.trained_task)
        self.step_loss['alphas'].append(curr_learner.alpha)
        curr_learner.reset_train_task()

    def add_mask_curr_info(self, curr_learner):
        for task_name, prob in curr_learner.task_mlm_prob.items():
            self.step_loss[f'{task_name}_mlm_prob'].append(prob)

    def save_task_loss_to_csv(self):
        df = pd.DataFrame(self.step_loss)
        df.to_csv(self.save_path, index=False)
        print(f"Save task loss file to {self.save_path}")


class GameMultiTaskTrainer(Trainer):
    def __init__(self, *args, task_loss_df_path='', **kwargs):
        super().__init__(*args, **kwargs)
        self.tasks_loss_tracker = TasksLossTracker(save_path=task_loss_df_path)

    def train(self, model_path: Optional[str] = None,
              trial: Union["optuna.Trial", Dict[str, Any]] = None,
              use_time_embed=0, compute_total_flos=True,
              find_unused_parameters=True,
              seperate_design_id=False,
              task_loss_df_path='',
              is_curriculum_task=0,
              is_curriculum_within_task=0,
              curr_learner=None,
              verbose=1
              ):

        """
        Main training entry point.

        Args:
            model_path (:obj:`str`, `optional`):
                Local path to the model if the model to train has been instantiated from a local path. If present,
                training will resume from the optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
        """

        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(self.args.seed)
            model = self.model_init()
            self.model = model.to(self.args.device)

            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            t_total = int(num_update_steps_per_epoch * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs
            self.args.max_steps = t_total

        self.create_optimizer_and_scheduler(num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (
                model_path is not None
                and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
                and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            self.optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            self.lr_scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model

        if self.args.fp16 and _use_apex:
            from transformers.file_utils import is_apex_available
            if is_apex_available():
                from apex import amp
            model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=self.args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=find_unused_parameters,
            )

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", self.args.to_json_string())
            self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        # Train!
        total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
        )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        print(f"Set compute_total_flos to {compute_total_flos}")
        self.global_step = 0
        self.epoch = 0
        self.total_flos = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split(os.path.sep)[0])
                self.total_flos = getattr(model.config, "total_flos", 0)

                epochs_trained = self.global_step // num_update_steps_per_epoch
                steps_trained_in_current_epoch = self.global_step % (num_update_steps_per_epoch)

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Continuing training from %d non-embedding floating-point operations", self.total_flos)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                self.total_flos = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = torch.tensor(0.0).to(self.args.device)
        logging_loss_scalar = 0.0
        model.zero_grad()
        disable_tqdm = self.args.disable_tqdm or not self.is_local_process_zero()
        train_pbar = trange(epochs_trained, int(np.ceil(num_train_epochs)), desc="Epoch", disable=disable_tqdm)
        curr_steps = 0

        for epoch in range(epochs_trained, int(np.ceil(num_train_epochs))):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            epoch_pbar = tqdm(epoch_iterator, desc="Iteration", disable=disable_tqdm)
            for step, inputs in enumerate(epoch_iterator):

                assert use_time_embed

                all_tasks = ['task0', 'task1', 'task2', 'task3', 'task4']
                curriculum_task_valid = False  # At least one task is valid for Curriculum Learning

                task_losses = []
                for task_name in all_tasks:
                    task_loss_gradient = get_target_loss(task_name,
                                                         inputs,
                                                         self.training_step,
                                                         model,
                                                         is_loss_backward=False if is_curriculum_task else True)
                    task_loss = float(task_loss_gradient)
                    task_losses.append((task_name, task_loss))
                    if is_curriculum_task:
                        if task_loss < curr_learner.alpha:
                            task_loss_gradient.backward()
                            tr_loss += task_loss_gradient.detach()
                            if verbose >= 2:
                                print(f"ADD Task-{task_name} loss-{task_loss} < {curr_learner.alpha}, backwarded")
                            curriculum_task_valid = True
                            curr_learner.add_train_task(task_name)
                        else:
                            if verbose >= 2:
                                print(f"Skip Task-{task_name} because loss {task_loss} > {curr_learner.alpha}")
                    elif is_curriculum_within_task:
                        tr_loss += task_loss_gradient.detach()
                        curr_learner.add_task_loss(task_name, task_loss)
                    else:
                        tr_loss += task_loss_gradient.detach()
                    self.tasks_loss_tracker.add_batch_loss(task_name, task_loss)

                if is_curriculum_task:
                    curr_learner.update_step()
                    curr_learner.adjust_alpha()
                    if not curriculum_task_valid:
                        task_losses = sorted(task_losses, key=lambda x: x[1])
                        task_name_lowest_loss = task_losses[0][0]
                        task_loss_gradient = get_target_loss(task_name_lowest_loss,
                                                             inputs,
                                                             self.training_step,
                                                             model,
                                                             is_loss_backward=False if is_curriculum_task else True)
                        task_loss_gradient.backward()
                        tr_loss += task_loss_gradient.detach()
                        if verbose >= 2:
                            print(f"ADD Task-{task_name_lowest_loss} loss, backwarded loss: {task_loss_gradient}")
                elif is_curriculum_within_task:
                    curr_learner.update_step()
                    curr_learner.adjust_mlm_prob(verbose=verbose)

                if compute_total_flos:
                    self.total_flos += self.floating_point_ops(inputs)
                else:
                    self.total_flos = None

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        len(epoch_iterator) <= self.args.gradient_accumulation_steps
                        and (step + 1) == len(epoch_iterator)
                ):
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    elif self.args.fp16 and _use_apex:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    if self.args.fp16 and _use_native_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.lr_scheduler.step()
                    model.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                            self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs: Dict[str, float] = {}
                        tr_loss_scalar = tr_loss.item()
                        logs["loss"] = (tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            self.lr_scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else self.lr_scheduler.get_lr()[0]
                        )
                        logging_loss_scalar = tr_loss_scalar
                        self.log(logs)
                        self.tasks_loss_tracker.compute_batch_loss()
                        if is_curriculum_task:
                            self.tasks_loss_tracker.add_curr_info(curr_learner)
                        elif is_curriculum_within_task:
                            self.tasks_loss_tracker.add_mask_curr_info(curr_learner)

                    if self.args.evaluate_during_training and self.global_step % self.args.eval_steps == 0:
                        metrics = self.evaluate()
                        self._report_to_hp_search(trial, epoch, metrics)

                    if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                        # In all cases (even distributed/parallel), self.model is always a reference
                        # to the model we want to save.
                        if hasattr(model, "module"):
                            assert (
                                    model.module is self.model
                            ), f"Module {model.module} should be a reference to self.model"
                        else:
                            assert model is self.model, f"Model {model} should be a reference to self.model"
                        # Save model checkpoint
                        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}"
                        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)

                        self.save_model(output_dir)

                        if self.is_world_process_zero():
                            self._rotate_checkpoints(use_mtime=True)

                        if self.is_world_process_zero():
                            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

                epoch_pbar.update(1)
                if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                    break
            epoch_pbar.close()
            train_pbar.update(1)
            if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                break

        train_pbar.close()
        if self.tb_writer:
            self.tb_writer.close()
        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        self.tasks_loss_tracker.save_task_loss_to_csv()

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss.item() / self.global_step)
