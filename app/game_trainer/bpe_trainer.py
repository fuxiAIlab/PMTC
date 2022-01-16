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

from app_utils import get_target_loss

logger = logging.get_logger(__name__)

_use_native_amp = False
_use_apex = False


def _update_posterior_prob(model,
                           inputs,
                           update_tensor,
                           alpha,
                           step_i,
                           epoch,
                           verbose=True,
                           logit_outputs=None):
    with torch.set_grad_enabled(False):
        model.eval()
        # dynamic_mask_predict_prob.prob_tensor
        # if step_i % update_step == 0:
        if logit_outputs is None:
            model_output = model(**inputs)
            logit_outputs = model_output[1]  # batch_size x max_seq_len x dim (50000)

        logit_outputs = logit_outputs.to('cpu')
        T = 1.0  # temperature
        prob_outputs = torch.softmax(logit_outputs / T, dim=2)  # batch_size x max_seq_len x dim
        batch_input_labels = inputs['labels']  # batch_size x max_seq_len
        # batch_masked_label_indices = torch.where(input_labels != -100)  # Tuple, len: batch_size,

        vocab_size = prob_outputs.shape[-1]
        new_predict_probs = []
        for i, input_labels in enumerate(batch_input_labels):
            masked_labels = input_labels[input_labels != -100]
            masked_label_indices = torch.where(input_labels != -100)[0]
            new_predict_prob = torch.zeros((vocab_size,))
            new_predict_prob[masked_labels] = prob_outputs[i][masked_label_indices, masked_labels]
            new_predict_probs.append(new_predict_prob)

        new_predict_probs = torch.stack(new_predict_probs)
        new_predict_prob = torch.mean(new_predict_probs, dim=0)
        update_vocab_indices = torch.where(new_predict_prob != 0)

        if update_vocab_indices[0].shape[0] != 0:
            old_predict_prob = update_tensor[update_vocab_indices]
            updated_predict_prob = alpha * new_predict_prob[update_vocab_indices] \
                                   + (1 - alpha) * old_predict_prob
            update_tensor[update_vocab_indices] = updated_predict_prob

        if step_i % 100 == 0 and verbose:
            print(
                f"epoch-{epoch}, Step-{step_i}, [Update dynamic_mask_predict_prob]"
                f" Max prob: {torch.max(update_tensor)}, "
                f" Min prob: {torch.min(update_tensor)}, "
                f" Avg prob: {torch.mean(update_tensor)}.")

        # # To check at least one token is masked
        # if batch_masked_label_indices[0].size()[0] != 0:
        #
        #
        #     for batch_i, masked_label_indices in enumerate(batch_masked_label_indices):
        #         # (length of masked seq, ), each element is label vocab index, element max value = vocab max size
        #         # (e.g. 50000)
        #         masked_label = input_labels[batch_i][masked_label_indices]
        #         ipdb.set_trace()

        # # TODO, temp debug
        # try:
        #     label_vocab_indices = input_labels[masked_label_indices]  # number_of_label_index,
        #     masked_label_probs = prob_outputs[masked_label_indices]  # number_of_label_index x vocab_size
        #     masked_label_mask = torch.zeros_like(masked_label_probs)
        #     masked_label_mask.scatter_(1, label_vocab_indices.unsqueeze(1).to(masked_label_mask.device), 1)
        #     masked_label_mask = masked_label_mask.bool()  # number_of_label_index x vocab_size
        #     new_predict_prob = masked_label_probs[masked_label_mask]
        #     old_predict_prob = update_tensor[label_vocab_indices]
        #     assert old_predict_prob.shape == new_predict_prob.shape
        #     updated_predict_prob = alpha * new_predict_prob + (1 - alpha) * old_predict_prob
        #     update_tensor[label_vocab_indices] = updated_predict_prob
        #     ipdb.set_trace()
        #
        #     if step_i % 100 == 0 and verbose:
        #         print(
        #             f"epoch-{epoch}, Step-{step_i}, [Update dynamic_mask_predict_prob]"
        #             f" Max prob: {torch.max(update_tensor)}, "
        #             f" Min prob: {torch.min(update_tensor)}, "
        #             f" Avg prob: {torch.mean(update_tensor)}.")
        # except:
        #     ipdb.set_trace()
        model.train()
    return logit_outputs


class BpeTasksLossTracker():
    def __init__(self, save_path=''):
        self.result_dict = {'gradient_acc_step_loss': [], 'mlm_prob': [], 'epoch': [], 'lr': [],
                            'softmax_t': [], 'part_prob': []}
        self.save_path = save_path
        self.step_i = 0

    def add_ga_step_loss(self, loss_value, mlm_prob, epoch_i, lr, softmax_t, part_prob=None):
        self.result_dict['gradient_acc_step_loss'].append(loss_value)
        self.result_dict['mlm_prob'].append(mlm_prob)
        self.result_dict['epoch'].append(epoch_i)
        self.result_dict['lr'].append(lr)
        self.result_dict['softmax_t'].append(softmax_t)
        if part_prob:
            self.result_dict['part_prob'].append(part_prob)
        self.step_i += 1

    def print_training_progress(self, total_step, epoch_pbar):
        epoch_pbar.set_description(f"{100 * (self.step_i / total_step):.2f} % training complete")

    def save_task_loss_to_csv(self):

        part_probs = self.result_dict.get('part_prob', [])
        keys = ['epoch', 'gradient_acc_step_loss', 'mlm_prob', 'lr', 'softmax_t', 'part_prob']
        if not part_probs:
            if 'part_prob' in self.result_dict:
                self.result_dict.pop('part_prob')
            keys.remove('part_prob')
        df = pd.DataFrame(self.result_dict)
        df = df[keys]
        df.to_csv(self.save_path, index=False)
        print(f"Save task loss file to {os.path.abspath(self.save_path)}")


class BpeTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.loss_tracker = kwargs['loss_tracker']
        if 'loss_tracker' in kwargs:
            kwargs.pop('loss_tracker')
        super().__init__(*args, **kwargs)

    def _save_model_etc(self, output_dir):
        self.save_model(output_dir)

        if self.is_world_process_zero():
            self._rotate_checkpoints(use_mtime=True)

        if self.is_world_process_zero():
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

    def train(self, model_path: Optional[str] = None,
              trial: Union["optuna.Trial", Dict[str, Any]] = None,
              use_time_embed=0, compute_total_flos=True,
              find_unused_parameters=True,
              is_curriculum_within_task=0,
              curr_learner=None,
              verbose=1,
              epoch_i=None,
              masker_recorder=None,
              dynamic_mask_update_alpha=0.5,
              mask_predict_prob_update_step=4,
              mask_type='normal',
              max_epoch=None,
              snapshot_gap=1000,
              shuffle_epoch=False,
              total_training_steps=0
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
            set_seed(self.args.seed + epoch_i)
            model = self.model_init()
            self.model = model.to(self.args.device)
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # TODO，目前就先不改了，因为之前跑的模型这里都没有改
        if shuffle_epoch:
            set_seed(self.args.seed + epoch_i)  # TODO, Set Seed to get different training order in each epoch

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)

        # if self.args.max_steps > 0:
        #     t_total = self.args.max_steps
        #     num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
        #         self.args.max_steps % num_update_steps_per_epoch > 0
        #     )
        # else:
        t_total = int(num_update_steps_per_epoch * self.args.num_train_epochs)
        num_train_epochs = self.args.num_train_epochs
        self.args.max_steps = t_total

        if epoch_i == 0:
            self.create_optimizer_and_scheduler(num_training_steps=total_training_steps)

            # overwrite lr scheduler
            from transformers import get_constant_schedule_with_warmup
            self.lr_scheduler = get_constant_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_steps)

        else:
            assert self.optimizer is not None
            assert self.lr_scheduler is not None

        # TODO: Disable warmup except for the first epoch
        model = self.model

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

        if masker_recorder.record_snapshot:
            print("[Record Snapshot mode], Set alpha to 0.5!")
            dynamic_mask_update_alpha = 0.5

        for epoch in range(epochs_trained, int(np.ceil(num_train_epochs))):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            gradient_acc_loss = 0
            epoch_pbar = tqdm(epoch_iterator, desc="Iteration", disable=disable_tqdm)

            # self.args.train_batch_size /
            steps_per_epoch = len(epoch_iterator) / self.args.gradient_accumulation_steps
            print(f"steps_per_epoch: {steps_per_epoch}")

            for step_i, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    epoch_pbar.update(1)
                    continue

                if not use_time_embed:
                    if 'timestamps' in inputs:
                        inputs.pop('timestamps')

                logit_outputs = None
                if masker_recorder.train_step != 0 and step_i % mask_predict_prob_update_step == 0:

                    # update token freq
                    input_labels = inputs['labels']
                    for input_label in input_labels:
                        masked_label_indices = input_label[input_label != -100]
                        masker_recorder.token_freq[masked_label_indices] += 1
                        masker_recorder.total_mask_label_count += len(masked_label_indices)

                    if mask_type in {'posterior_prob',
                                     'posterior_prob_with_tf_idf_warmup',
                                     'lowest_prob',
                                     'part_prob',
                                     'part_prob_linear_increase'
                                     } or masker_recorder.record_snapshot:
                        logit_outputs = _update_posterior_prob(model,
                                                               inputs,
                                                               masker_recorder.prob_tensor,
                                                               dynamic_mask_update_alpha,
                                                               step_i,
                                                               epoch)

                    if epoch_i == max_epoch - 1:
                        _ = _update_posterior_prob(model,
                                                   inputs,
                                                   masker_recorder.token_last_predict_prob,
                                                   1.0,
                                                   step_i,
                                                   epoch,
                                                   logit_outputs=logit_outputs)
                        # record token mask count in the last epoch
                        # Keep track of the frequency of the masked tokens

                step_loss = self.training_step(model, inputs)
                tr_loss += step_loss
                gradient_acc_loss += float(step_loss)

                # # -----------------------------------------------
                # # temp
                # # -----------------------------------------------
                # model_output = model(**inputs, output_hidden_states=True)[2][-2].detach()
                # mean_pool_output = torch.mean(model_output, dim=1)
                # from sklearn.metrics.pairwise import cosine_similarity
                # cos_sim = cosine_similarity(mean_pool_output[0].unsqueeze(0).detach().cpu(),
                #                             mean_pool_output[1].unsqueeze(0).detach().cpu())
                # print(f"cos_sim: {cos_sim}")
                # # -----------------------------------------------

                if compute_total_flos:
                    self.total_flos += self.floating_point_ops(inputs)
                else:
                    self.total_flos = None

                if (step_i + 1) % self.args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        len(epoch_iterator) <= self.args.gradient_accumulation_steps
                        and (step_i + 1) == len(epoch_iterator)
                ):
                    self.optimizer.step()
                    model.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step_i + 1) / len(epoch_iterator)

                    # Update mask recorder
                    masker_recorder.train_step += 1

                    # Record Snapshots
                    if masker_recorder.record_snapshot:
                        if masker_recorder.train_step % snapshot_gap == 0:
                            masker_recorder.add_snapshot(masker_recorder.train_step, masker_recorder.prob_tensor)
                            print(f"[Record snapshot] step_i: {masker_recorder.train_step},"
                                  f" total recored: {len(masker_recorder.prob_snapshots)}")

                    # update softmax_t
                    self.data_collator.adjust_mask_softmax_t(masker_recorder.train_step)
                    self.data_collator.adjust_part_prob_percent(masker_recorder.train_step)

                    # update loss tracker
                    self.loss_tracker.add_ga_step_loss(gradient_acc_loss,
                                                       None,
                                                       epoch_i,
                                                       self.lr_scheduler.get_last_lr()[0],
                                                       self.data_collator.mask_softmax_t,
                                                       part_prob=self.data_collator.part_prob_percent
                                                       )
                    self.loss_tracker.print_training_progress(total_training_steps, epoch_pbar)

                    self.lr_scheduler.step()

                    # reset gradient acc loss
                    gradient_acc_loss = 0

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                            self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs: Dict[str, float] = {}
                        tr_loss_scalar = tr_loss.item()
                        logs["loss"] = (tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
                        # print(f"logs_loss: {logs['loss']}, step_i: {step_i}")

                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            self.lr_scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else self.lr_scheduler.get_lr()[0]
                        )
                        logging_loss_scalar = tr_loss_scalar
                        self.log(logs)

                        # self.tasks_loss_tracker.compute_batch_loss()

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

                        self._save_model_etc(output_dir)

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

        if self.args.local_rank in {-1, 0}:
            self.loss_tracker.save_task_loss_to_csv()

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss.item() / self.global_step)
