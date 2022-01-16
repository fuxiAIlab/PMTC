import copy
import ipdb
import torch
import random
import collections
import numpy as np
from pytorch_lightning.callbacks import Callback
from .utils import is_same_state_dict
from .utils import load_save_json
from .train_metric import MacroF1Metric
from .train_metric import NDCGMetric


def _add_best_embedder_parameters(embedder, best_embedder_parameters):
    # get model name & pretrain task
    model_name = embedder.model_name
    pretrain_task = embedder.pretrain_task

    # (1.) add Bert parameters
    best_embedder_parameters.append({'model': embedder.model, 'model_name': model_name})
    # (1.1) add tfidf_adapter
    if hasattr(embedder, 'tfidf_adapter'):
        best_embedder_parameters.append({'model': embedder.tfidf_adapter, 'model_name': model_name + '_tfidf_adapter'})
    # (1.2) add timegap_adapter
    if hasattr(embedder, 'timegap_adapter'):
        best_embedder_parameters.append(
            {'model': embedder.timegap_adapter, 'model_name': model_name + '_timegap_adapter'})


class GameTrainerCallback(Callback):
    """
    目前这个类写的不是很好，因为这里不同的metric其实是和模型绑定的，不应该用同一个callback来完成，我觉得可能不同的模型用多个
    callback会更清晰一点
    """

    def on_init_start(self, trainer):
        trainer.train_losses = []
        trainer.val_losses = []
        trainer.test_loss = None
        trainer.test_metric_value = None
        trainer.test_metric_value_shuffle = None
        trainer.best_state_dict = None
        trainer.bert_best_state_dict = None
        trainer.val_metric_values = []
        print('Starting to init trainer!')

    def on_init_end(self, trainer):
        print('trainer is init now')

    def _is_select_model_by_val_metric(self, pl_module):
        if hasattr(pl_module, 'is_select_model_by_val_metric'):
            is_select_model_by_val_metric = pl_module.is_select_model_by_val_metric
        else:
            is_select_model_by_val_metric = False
        return is_select_model_by_val_metric

    def on_train_start(self, trainer, pl_module):
        # Add model parameters
        if trainer.is_finetune_bert:
            trainer.best_embedder_parameters = []
            _add_best_embedder_parameters(trainer.embedder, trainer.best_embedder_parameters)
            if hasattr(trainer, 'union_embedder'):
                if id(trainer.embedder) != id(trainer.union_embedder):
                    _add_best_embedder_parameters(trainer.union_embedder, trainer.best_embedder_parameters)
        else:
            trainer.best_embedder_parameters = None

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        trainer.train_losses.append(float(trainer.callback_metrics['train_loss']))

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.is_finetune_bert:
            trainer.embedder.model.train()
            print(f"[FineTune Mode] Model is set to TRAIN before train!")

    def on_validation_start(self, trainer, pl_module):
        if trainer.is_finetune_bert:
            trainer.embedder.model.eval()
            print(f"[FineTune Mode] Model is set to EVAL before val!")

    def _interpret_test_results(self, trainer, pl_module):
        trainer.interpreter.load_test_model(pl_module)

        # collect test data
        test_dataloader = iter(trainer.test_dataloaders[0])
        length_dataloader = iter(trainer.test_dataloaders[0])

        test_bpe_seqs = []
        test_samples = []
        test_labels = []

        sample_lengths = []
        for batch_sample in length_dataloader:
            sample_lengths.append(batch_sample[0][0].shape[1])
        pad_length = max(sample_lengths)

        # load cn_char translation dict
        tokenizer = trainer.target_task.embedder.tokenizer
        tokenizer_re_vocab = {v: k for k, v in tokenizer.get_vocab().items()}
        game_id_cn_char_dict = load_save_json('../../static/game_id_cn_char.dict', 'load')
        unk_cn_char = game_id_cn_char_dict['[UNK]']
        re_game_id_cn_char_dict = {y: x for x, y in game_id_cn_char_dict.items()}

        for batch_sample in test_dataloader:
            batch_sample_x = batch_sample[0]
            batch_y = batch_sample[1]
            test_labels.extend(batch_y)
            batch_embedding, _, batch_raw_seq = batch_sample_x
            for x, raw_seq in zip(batch_embedding, batch_raw_seq):
                raw_seq_wo_time = [x for i, x in enumerate(raw_seq) if (i + 1) % 3 != 0]
                raw_seq_cn = [game_id_cn_char_dict.get(x, unk_cn_char) for x in raw_seq_wo_time]
                bpe_seq_id = tokenizer.encode(''.join(raw_seq_cn)).ids
                bpe_seq_cn_char = [tokenizer_re_vocab[x] for x in bpe_seq_id]
                bpe_trans_back = [[re_game_id_cn_char_dict.get(y, re_game_id_cn_char_dict[unk_cn_char]) for y in x]
                                  for x in bpe_seq_cn_char]
                test_bpe_seqs.append(bpe_trans_back)
                pad_x = torch.zeros((pad_length, x.shape[-1]))
                pad_x[:len(x)] = x
                test_samples.append(pad_x)
        test_samples = torch.stack(test_samples)
        trainer.interpreter.compute_attribute(test_samples, test_labels, test_bpe_seqs)

    def on_test_start(self, trainer, pl_module):

        # Load best parameters
        if trainer.is_finetune_bert:
            self._load_train_best_state_dicts(trainer)
        pl_module.load_state_dict(trainer.best_state_dict)
        print("[Before Test] Load best state dict done")

        # Set embedder to train mode
        if trainer.is_finetune_bert:
            trainer.embedder.model.eval()
            print(f"[FineTune Mode] Model is set to EVAL before test!")

        if trainer.interpreter is not None:
            self._interpret_test_results(trainer, pl_module)

    def _update_metric_values(self, target_metric, outputs):
        # TODO, 这一块写的不是特别好，目前是靠metric和模型绑定的，也就是说用相同metric的模型的output必须也相同
        if isinstance(target_metric, MacroF1Metric):
            _, (true_labels, predict_labels) = outputs
            target_metric.extend_ys(predict_labels, true_labels)
        elif isinstance(target_metric, NDCGMetric):
            _, (role_id_dses, true_probs, output_probs, random_probs) = outputs
            target_metric.update_y(role_id_dses, true_probs, output_probs, random_probs)
        else:
            raise NotImplementedError

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self._is_select_model_by_val_metric(pl_module):
            self._update_metric_values(pl_module.val_metric, outputs)

    def _update_trainer_best_state_dicts(self, trainer, is_epoch_end=False):
        for model_dict in trainer.best_embedder_parameters:
            if is_epoch_end:
                previous_best_params = model_dict.get('best_params', None)
                if previous_best_params is None:
                    model_dict['best_params'] = model_dict['model'].state_dict()
            else:
                model_dict['best_params'] = model_dict['model'].state_dict()

    def _load_train_best_state_dicts(self, trainer):
        print("\n")
        for model_dict in trainer.best_embedder_parameters:
            model = model_dict['model']
            model_name = model_dict['model_name']
            best_params = model_dict['best_params']
            model.load_state_dict(best_params)
            print(f"[Load Best params] Model-{model_name} load best params done!")

    def on_validation_end(self, trainer, pl_module):
        if not trainer.running_sanity_check:
            global_step = trainer.global_step

            if self._is_select_model_by_val_metric(pl_module):
                target_metric = pl_module.val_metric
                val_metric_value = target_metric.compute_metric()

                if not trainer.val_metric_values:
                    trainer.val_metric_values.append(val_metric_value)

                # TODO, 这里的Test Metric都是越大越好的，以后越小越好的要注意一下
                if len(trainer.val_metric_values) >= 1:
                    if len(trainer.val_metric_values) == 1:
                        previous_best = float('-inf')
                    else:
                        previous_best = max(trainer.val_metric_values)
                    if val_metric_value >= previous_best:
                        if trainer.is_finetune_bert:
                            self._update_trainer_best_state_dicts(trainer)
                        trainer.best_state_dict = pl_module.state_dict()
                        print(
                            f"[Find Best Model] global_step: {global_step},"
                            f" val {target_metric.name} {val_metric_value} equal or better than {previous_best}")
                    else:
                        print(
                            f"[Val result] global_step: {global_step},"
                            f" val {target_metric.name} {val_metric_value}, best score: {previous_best}")
                trainer.val_metric_values.append(val_metric_value)
            else:
                val_loss = float(trainer.callback_metrics['val_loss'])
                if not trainer.val_losses:
                    trainer.val_losses.append(val_loss)
                if len(trainer.val_losses) >= 1:
                    if len(trainer.val_losses) == 1:
                        previous_best = float('inf')
                    else:
                        previous_best = min(trainer.val_losses)
                    if val_loss < previous_best:
                        if trainer.is_finetune_bert:
                            self._update_trainer_best_state_dicts(trainer)
                        trainer.best_state_dict = pl_module.state_dict()
                        print(
                            f"[Find Best Model] global_step: {global_step}, val loss {val_loss} better than {previous_best}")
                    else:
                        print(
                            f"[VAL] global_step: {global_step}, val_loss: {val_loss}, best val loss {min(trainer.val_losses)}")
                trainer.val_losses.append(val_loss)
        else:
            print("At the End of the validation sanity check")

    def on_train_end(self, trainer, pl_module):
        # Use the last epoch parameters if best_state_dict is None
        if trainer.best_state_dict is None:
            trainer.best_state_dict = pl_module.state_dict()

        if trainer.is_finetune_bert:
            self._update_trainer_best_state_dicts(trainer, is_epoch_end=True)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self._update_metric_values(pl_module.test_metric, outputs)

    # def on_test_epoch_start(self, trainer, pl_module):
    #     if trainer.is_finetune_bert:
    #         self._load_train_best_state_dicts(trainer)
    #     pl_module.load_state_dict(trainer.best_state_dict)
    #     print("[Before Test] Load best state dict done")

    def on_test_end(self, trainer, pl_module):
        test_metric = pl_module.test_metric
        if isinstance(test_metric, MacroF1Metric):
            macro_f1, shuffle_macro_f1 = pl_module.test_metric.compute_metric_etc()
            print(f"Test, shuffle f1: {shuffle_macro_f1}, model f1: {macro_f1}")
            trainer.test_metric_value = macro_f1
            trainer.test_metric_value_shuffle = shuffle_macro_f1
        elif isinstance(test_metric, NDCGMetric):
            ndcg_score, ndcg_score_random, ndcg_scores = pl_module.test_metric.compute_metric_etc()
            print(
                f"Test, "
                f"random ndcg_score: {ndcg_score_random}, "
                f"ndcg_score: {ndcg_score},"
                f" max-{np.max(ndcg_scores)},"
                f" min-{np.min(ndcg_scores)},"
                f" std-{np.std(ndcg_scores)}")
            trainer.test_metric_value = ndcg_score
            trainer.test_metric_value_shuffle = ndcg_score_random
        else:
            raise NotImplementedError

        # some post-processing
        # Discard the first val loss
        assert is_same_state_dict(pl_module.state_dict(), trainer.best_state_dict)
        trainer.val_losses = trainer.val_losses[1:]
        trainer.test_loss = float(trainer.callback_metrics['test_loss'])
