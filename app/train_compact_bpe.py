"""
This is a compact training script for game bert
"""

import logging
import math
import glob
import os
import time
import ipdb
import torch
import subprocess
import sys

sys.path.append('..')

from app_utils import load_save_json
from app_utils import change_trainer_output_dir
from app_utils import hdf5_load_dataset
from app_utils import check_batch_hdf5_dataset_size
from app_utils import get_all_indices
from app_utils import set_randseed
from pytorch_lightning import seed_everything
from args_utils import ModelArguments, DataTrainingArguments
from game_dataset.bpe_dataset import get_dataset_bpe
from game_trainer.curriculum_learner import BpeAdjustMaskCurriculumLearner
from game_trainer.bpe_trainer import BpeTrainer
from game_trainer.bpe_trainer import BpeTasksLossTracker
from game_trainer.bpe_data_collator import BpeDataCollatorForLanguageModeling
from pjs_bert_model.bert_model import BertForMaskedLMTimeEmbed
from pjs_bert_model.mask_with_dynamic_prob import DynamicMaskerRecorder
from pjs_bert_model.bert_model_add_mean_pool import BertForMaskedLMMeanPool

# Init logger
logger = logging.getLogger(__name__)

from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers import (
    CONFIG_MAPPING,
    AutoModelWithLMHead)


def init_etc(training_args, data_args, model_args):
    # (1.) Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # (2.) overwrite output dir
    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # (3.) set random seed
    set_seed(training_args.seed)
    set_randseed(1)

    # (4.) remove cache feature file
    if training_args.local_rank in [-1, 0]:
        feature_save_dir = os.path.dirname(data_args.h5_data_file_path)
        cached_file_paths = glob.glob(os.path.join(feature_save_dir, 'cached_*'))
        for path in cached_file_paths:
            os.remove(path)
            print(f"Remove cached path {path}")

    # (5.) init print
    print(f"[H5 data path]: {data_args.h5_data_file_path}")

    # (6.) change new output dir
    new_output_dir = change_trainer_output_dir(training_args.output_dir, model_args, logger, data_args)
    return new_output_dir


def update_model_config(config, data_args, model_args):
    # add gate for timestamp for design id
    if data_args.is_gate_design_id:
        config.is_gate_design_id = True
    if data_args.is_gate_timestamp:
        config.is_gate_timestamp = True

    if data_args.is_timestamp_rnn:
        config.is_timestamp_rnn = True

    # load default config for each model
    from game_config.game_model_config import game_config, longformer_config, transformer_xl_config, albert_config, \
        reformer_config
    find_unused_parameters = True
    if model_args.model_type in {'bert', 'longformer'}:
        for attribute, value in game_config.items():
            setattr(config, attribute, value)
            print(f"[CHANGE CONFIG] set {attribute} to {value}")

        if model_args.model_type == 'longformer':
            for attribute, value in longformer_config.items():
                setattr(config, attribute, value)
                print(f"[CHANGE CONFIG] set {attribute} to {value}")
            config.max_position_embeddings -= 2  # For roberta emebdding offset
            assert not model_args.use_time_embed

    elif model_args.model_type in {'transfo-xl'}:
        assert not model_args.use_time_embed
        for attribute, value in transformer_xl_config.items():
            setattr(config, attribute, value)
            print(f"[CHANGE CONFIG] set {attribute} to {value}")

    elif model_args.model_type in {'albert'}:
        assert not model_args.use_time_embed
        for attribute, value in albert_config.items():
            setattr(config, attribute, value)
            print(f"[CHANGE CONFIG] set {attribute} to {value}")

    elif model_args.model_type == 'reformer':
        assert not model_args.use_time_embed
        for attribute, value in reformer_config.items():
            setattr(config, attribute, value)
            print(f"[CHANGE CONFIG] set {attribute} to {value}")
        config.axial_pos_shape = [reformer_config.axial_pos_value,
                                  int(data_args.block_size / reformer_config.axial_pos_value)]
        config.is_decoder = True
        assert len(config.attn_layers) == config.num_hidden_layers
        find_unused_parameters = False

    # set max_position_embeddings
    if model_args.model_type == 'longformer':
        config.max_position_embeddings = data_args.block_size + 2
    else:
        config.max_position_embeddings = data_args.block_size
    print(f"Set max position embedding to {data_args.block_size}")

    # update training_args
    if data_args.is_task0:
        config.is_task0 = True
    else:
        config.is_task0 = False
    if data_args.is_task1:
        config.is_task1 = True
    else:
        config.is_task1 = False
    if data_args.is_task2:
        config.is_task2 = True
    else:
        config.is_task2 = False
    if data_args.is_task3:
        config.is_task3 = True
    else:
        config.is_task3 = False
    if data_args.is_task4:
        config.is_task4 = True
    else:
        config.is_task4 = False

    return config, find_unused_parameters


def init_tokenizer(model_args):
    from tokenizers import Tokenizer
    from game_tokenizer.game_tokenizer import create_func1, create_func2
    logging.info(f"Load tokenizer from {model_args.bpe_tokenizer_path}")
    tokenizer = Tokenizer.from_file(model_args.bpe_tokenizer_path)
    tokenizer.max_len = len(tokenizer.get_vocab())
    tokenizer.cls_token_id = 0
    tokenizer.pad_token_id = 1
    tokenizer.sep_token_id = 2
    tokenizer.unk_token_id = 3
    tokenizer.mask_token_id = 4
    tokenizer.get_special_tokens_mask = create_func1(tokenizer.pad_token_id, tokenizer.cls_token_id)
    tokenizer.added_tokens_encoder = {}
    tokenizer.convert_tokens_to_ids = create_func2(tokenizer.added_tokens_encoder, tokenizer.mask_token_id)
    tokenizer.mask_token = '[MASK]'
    tokenizer._pad_token = '[PAD]'
    is_decode_utf8 = True
    use_bpe = True
    return tokenizer, is_decode_utf8, use_bpe


def init_model(config, model_args, tokenizer):
    config.time_gap_max_size = model_args.max_time_gap
    config.use_sinusoidal = True
    config.vocab_size = tokenizer.max_len
    config.seperate_design_id = False

    if model_args.use_time_embed:
        model = BertForMaskedLMTimeEmbed(config)
        print(f"[TIME EMBED] Initialize time embedding BERT model!")
    elif model_args.add_mean_pool_to_train:
        model = BertForMaskedLMMeanPool(config)
        print(f"[TIME EMBED] Initialize BERT with Mean pooling modification")
    else:
        # TODO, temp change config
        logger.info("Training new model from scratch")
        # ipdb.set_trace()
        config.vocab_size = len(tokenizer.get_vocab())
        base_cutoff = int(config.vocab_size / 10)
        config.cutoffs = [base_cutoff, base_cutoff * 2, base_cutoff * 10]
        model = AutoModelWithLMHead.from_config(config)

    if tokenizer is not None:
        model.resize_token_embeddings(len(tokenizer.get_vocab()))
        print(f"Resize token embedding done, set to {len(tokenizer.get_vocab())}")

    # 在print模型的时候sinusoidal_time_embeddings会看不到，因为它不是pytorch的组件
    print(model)
    model_total_params = sum(p.numel() for p in model.parameters())
    print(f"model total params: {model_total_params}")

    return model


def save_meta_config(model_args, data_args, new_output_dir, data_collator, total_step):
    if model_args.use_white_space_tokenzier:
        meta_tokenizer = 'whitespace'
    elif model_args.use_bpe_tokenzier:
        meta_tokenizer = 'bpe'
    else:
        raise NotImplementedError

    meta_config = {'model': model_args.model_type,
                   'pretrain_task': data_args.pretrain_task,
                   'tokenizer': meta_tokenizer,
                   'is_time_embed': model_args.use_time_embed,
                   'use_sinusoidal': data_args.use_sinusoidal,
                   'task_curriculum_update_step': data_args.task_curriculum_update_step,
                   'is_curriculum_within_task': data_args.is_curriculum_within_task,
                   'last_mlm_prob': data_collator.mlm_probability,
                   'use_random_mlm_probability': data_args.use_random_mlm_probability,
                   'mlm_prob_min': data_args.mlm_prob_min,
                   'mlm_prob_max': data_args.mlm_prob_max,
                   'is_anti_curr': data_args.is_anti_curr,
                   'train_mean_pool': model_args.add_mean_pool_to_train,
                   'mask_type': data_args.mask_type,
                   'total_step': total_step,
                   'tf_idf_warmup_decay': data_args.tf_idf_warmup_decay,
                   'mask_softmax_t': data_args.mask_softmax_t,
                   'dynamic_mask_update_alpha': data_args.dynamic_mask_update_alpha
                   }
    meta_config_save_path = os.path.join(new_output_dir, 'meta_config.json')
    load_save_json(meta_config_save_path, 'save', data=meta_config)


def _translate_cn_label_to_raw_label(label_cn_char, re_game_id_cn_char_dict):
    if label_cn_char not in {'<pad>', '<s>', '</s>', '<unk>', '<mask>', '▁'}:
        raw_cn_strs = [x.replace('▁', '').replace('_', '') for x in label_cn_char]
        label_name = tuple(
            [re_game_id_cn_char_dict.get(cn_str, '') for cn_str in raw_cn_strs if cn_str])
    else:
        label_name = label_cn_char
    return label_name


def save_train_predict_prob(masker_recorder, tokenizer, new_output_dir, data_args):
    mask_type = data_args.mask_type
    import pandas as pd
    dynamic_mask_predict_prob_df = {'label': [],
                                    'label_index': [],
                                    'acc_prob': []}

    last_epoch_df = {
        'label': [],
        'label_index': [],
        'freq': [],
        'prob': []
    }

    game_id_cn_char_dict = load_save_json('../static/game_id_cn_char.dict', 'load')
    re_game_id_cn_char_dict = {y: x for x, y in game_id_cn_char_dict.items()}
    tokenizer_vocab = {v: k for k, v in tokenizer.get_vocab().items()}

    if mask_type in {'posterior_prob', 'posterior_prob_with_tf_idf_warmup'}:
        sorted_labels = torch.argsort(masker_recorder.prob_tensor, descending=True)
        for label_index in sorted_labels:
            label_cn_char = tokenizer_vocab[int(label_index)]
            if label_cn_char not in {'<pad>', '<cls>', '<sep>'}:
                raw_cn_strs = [x.replace('▁', '').replace('_', '') for x in label_cn_char]
                label_name = tuple(
                    [re_game_id_cn_char_dict.get(cn_str, '') for cn_str in raw_cn_strs if cn_str])
            else:
                label_name = label_cn_char
            dynamic_mask_predict_prob_df['label'].append(label_name)
            dynamic_mask_predict_prob_df['label_index'].append(int(label_index))
            dynamic_mask_predict_prob_df['acc_prob'].append(float(masker_recorder.prob_tensor[int(label_index)]))
        dynamic_mask_predict_prob_df = pd.DataFrame(dynamic_mask_predict_prob_df)
        dynamic_mask_predict_prob_df['mask_type'] = data_args.mask_type
        dynamic_mask_predict_prob_df['mask_softmax_t'] = data_args.mask_softmax_t
        dynamic_mask_predict_prob_df['mask_alpha'] = data_args.dynamic_mask_update_alpha
        dynamic_mask_save_path = os.path.join(new_output_dir, 'train_predict_prob.csv')
        dynamic_mask_predict_prob_df.to_csv(dynamic_mask_save_path, index=False)
        print(f"Save train predict prob to {dynamic_mask_save_path}")

    last_epoch_save_path = os.path.join(new_output_dir, 'train_last_epoch.csv')
    for label_index in range(len(tokenizer_vocab)):
        label_cn_char = tokenizer_vocab[int(label_index)]
        label_raw = _translate_cn_label_to_raw_label(label_cn_char, re_game_id_cn_char_dict)
        freq = int(masker_recorder.token_freq[label_index])
        prob = float(masker_recorder.token_last_predict_prob[label_index])
        last_epoch_df['label'].append(label_raw)
        last_epoch_df['label_index'].append(label_index)
        last_epoch_df['freq'].append(freq)
        last_epoch_df['prob'].append(prob)
    last_epoch_df = pd.DataFrame(last_epoch_df)
    last_epoch_df['mask_type'] = data_args.mask_type
    last_epoch_df['mask_softmax_t'] = data_args.mask_softmax_t
    last_epoch_df['mask_alpha'] = data_args.dynamic_mask_update_alpha
    last_epoch_df['total_mask_label_count'] = masker_recorder.total_mask_label_count
    last_epoch_df['overshoot_count'] = masker_recorder.overshoot_count

    last_epoch_df.to_csv(last_epoch_save_path, index=False)
    print(f"Save last epoch statistics to {last_epoch_save_path}")

    # Save tf_idf warmup prob
    if masker_recorder.tf_idf_warm_up_probs:
        tf_idf_warm_up_probs_save_path = os.path.join(new_output_dir, 'tf_idf_warm_up_probs.txt')
        with open(tf_idf_warm_up_probs_save_path, 'w') as f:
            for x in masker_recorder.tf_idf_warm_up_probs:
                f.write(str(x) + '\n')
        print(f"Save warm up statistics to {tf_idf_warm_up_probs_save_path}")

    # Save snapshots
    if masker_recorder.record_snapshot:
        # save prob snapshots
        snapshot_dfs = []
        snapshot_save_path = os.path.join(new_output_dir, 'prob_snapshots.csv')
        for step_i, snapshot_t in masker_recorder.prob_snapshots:
            snapshot_df = {'prob': snapshot_t.tolist(), 'index': list(range(len(snapshot_t)))}
            snapshot_df = pd.DataFrame(snapshot_df)
            snapshot_df['step'] = step_i
            snapshot_dfs.append(snapshot_df)
        snapshot_df = pd.concat(snapshot_dfs)
        snapshot_df['mask_type'] = data_args.mask_type
        snapshot_df['mask_softmax_t'] = data_args.mask_softmax_t
        snapshot_df['mask_alpha'] = data_args.dynamic_mask_update_alpha
        snapshot_df.to_csv(snapshot_save_path, index=False)
        print(f"Save snapshot df to {snapshot_save_path}, shape: {snapshot_df.shape}")

        # save mask distribution snapshots
        dis_snapshot_save_path = os.path.join(new_output_dir, 'mask_dis_snapshots.csv')
        dis_snap_shot_df = {'step': [], 'dis_snapshot': []}
        for step_i, dis_snapshot in masker_recorder.mask_distribution_snapshots:
            dis_snap_shot_df['step'].append(step_i)
            dis_snap_shot_df['dis_snapshot'].append(dis_snapshot)
        dis_snap_shot_df = pd.DataFrame(dis_snap_shot_df)
        dis_snap_shot_df['mask_type'] = data_args.mask_type
        dis_snap_shot_df['mask_softmax_t'] = data_args.mask_softmax_t
        dis_snap_shot_df['mask_alpha'] = data_args.dynamic_mask_update_alpha
        dis_snap_shot_df.to_csv(dis_snapshot_save_path, index=False)
        print(f"Save mask dis snapshot df to {dis_snapshot_save_path}, shape: {dis_snap_shot_df.shape}")

        # save step_sample_mask_distributions
        step_sample_mask_dis_save_path = os.path.join(new_output_dir, 'sample_mask_distribution.csv')
        step_sample_mask_distributions = masker_recorder.step_sample_mask_distributions
        step_sample_mask_distributions = pd.DataFrame(step_sample_mask_distributions)
        step_sample_mask_distributions.to_csv(step_sample_mask_dis_save_path, index=False)
        print(
            f"Save sample mask distribution to {step_sample_mask_dis_save_path}, "
            f"shape: {step_sample_mask_distributions.shape}")

        masker_recorder.reset_step_mask_probabilities()

    # Save mask raitos
    if data_args.is_record_mask_ratio:
        mask_ratio_save_path = os.path.join(new_output_dir, 'mask_ratios.txt')
        with open(mask_ratio_save_path, 'w') as f:
            for x in masker_recorder.mask_ratios:
                f.write(str(x) + '\n')
        print(f"Save mask ratios to {mask_ratio_save_path}")


def _get_total_steps(data_args, training_args, all_indices, step_size):
    epoch_total_steps = 0
    for large_batch_i, batch_train_size in enumerate(check_batch_hdf5_dataset_size(data_args.h5_data_file_path,
                                                                                   all_indices,
                                                                                   step_size,
                                                                                   large_batch=data_args.large_batch)):
        large_batch_step = math.floor(batch_train_size / training_args.train_batch_size
                                      / training_args.gradient_accumulation_steps)
        epoch_total_steps += large_batch_step
    total_steps = int(data_args.outer_epoch * epoch_total_steps)
    return total_steps


def parse_args():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    return model_args, data_args, training_args


def main():
    seed = 1
    # seed everything
    seed_everything(seed)

    # get cache dir
    host_id = subprocess.check_output('hostid').strip().decode('utf-8')
    embedding_cache_dir = ''
    read_npy_from_disk = False

    # (1.) parse args
    model_args, data_args, training_args = parse_args()

    # Some special treatment to args
    if data_args.pretrain_task == 'mlm':
        data_args.mlm = True
        data_args.clm = False
    elif data_args.pretrain_task == 'clm':
        data_args.mlm = False
        data_args.clm = True
    else:
        raise NotImplementedError

    # (2.) init other things
    new_output_dir = init_etc(training_args, data_args, model_args)

    # (3.) init config
    config = CONFIG_MAPPING[model_args.model_type]()
    logger.warning("You are instantiating a new config instance from scratch.")

    # (4.) modify config
    config, find_unused_parameters = update_model_config(config, data_args, model_args)

    # init masker_recorder
    masker_recorder = DynamicMaskerRecorder(data_args.mask_type,
                                            data_args.idf_path,
                                            record_snapshot=data_args.is_record_snapshot,
                                            is_record_mask_ratio=data_args.is_record_mask_ratio)

    # (5.) init tokenizer & data_collator
    tokenizer, is_decode_utf8, use_bpe = init_tokenizer(model_args)
    data_collator = BpeDataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        use_time_embed=model_args.use_time_embed,
        mlm_probability=data_args.mlm_probability,
        pretrain_task=data_args.pretrain_task,
        clm_sample_n=data_args.clm_sample_n,
        use_random_mlm_probability=data_args.use_random_mlm_probability,
        mlm_prob_min=data_args.mlm_prob_min,
        mlm_prob_max=data_args.mlm_prob_max,
        masker_recorder=masker_recorder,
        mask_type=data_args.mask_type,
        mask_softmax_t=data_args.mask_softmax_t,
        tf_idf_warmup_decay=data_args.tf_idf_warmup_decay,
        is_record_mask_ratio=data_args.is_record_mask_ratio,
        softmax_t_decay_mode=data_args.softmax_t_decay_mode,
        part_prob_percent=data_args.part_prob_percent,
        part_prob_range=(data_args.part_prob_min, data_args.part_prob_max)
    )
    data_collator.use_time_embed = False
    data_collator.seperate_design_id = False

    # (6.) init model
    model = init_model(config, model_args, tokenizer)

    # (7.) init mask curr learner
    if data_args.is_curriculum_within_task:
        curr_learner = BpeAdjustMaskCurriculumLearner(data_args.task_curriculum_update_step,
                                                      data_args.mlm_probability,
                                                      data_collator,
                                                      bpe_cr_update_coeff=data_args.bpe_cr_update_coeff,
                                                      bpe_cr_adjust_tol=data_args.bpe_cr_adjust_tol,
                                                      is_anti_curr=data_args.is_anti_curr
                                                      )
    else:
        curr_learner = None

    # (7.5) init loss tracker
    loss_tracker = BpeTasksLossTracker(save_path=os.path.join(new_output_dir, 'tasks_loss.csv'))

    # (7.) Below is the code for training
    # Get datasets for each round
    is_print_pad_ratio = False
    all_indices, total_num = get_all_indices(data_args.h5_data_file_path,
                                             model_args.debugN,
                                             is_print_pad_ratio=is_print_pad_ratio)
    step_size = math.ceil(total_num / data_args.large_batch)

    # ipdb> tokenizer_vocab[1]
    # '<pad>'
    # ipdb> tokenizer_vocab[0]
    # '<s>'
    # ipdb> tokenizer_vocab[2]
    # '</s>'
    # ipdb> tokenizer_vocab[3]
    # '<unk>'
    # ipdb> tokenizer_vocab[4]
    # '<mask>'
    # ipdb> tokenizer_vocab[5]
    # '▁'
    # tokenizer_vocab = {v: k for k, v in tokenizer.get_vocab().items()}
    optimizer, lr_scheduler = None, None

    # compute total steps
    total_training_steps = _get_total_steps(data_args,
                                            training_args,
                                            all_indices,
                                            step_size)
    print(f"[Total Step] Total step: {total_training_steps}")
    data_collator.total_training_step = total_training_steps
    masker_recorder.total_steps = total_training_steps
    seed_everything(seed)

    # Set Warmup steps
    warm_up_percent = 0.05
    warmup_steps = math.ceil(total_training_steps * warm_up_percent)
    warmup_steps = max(warmup_steps, 2)
    training_args.warmup_steps = warmup_steps
    print(f"[Set WARMUP Steps] set warmup steps to {warmup_steps}")

    for epoch_i in range(data_args.outer_epoch):
        for large_batch_i, large_batch_data in enumerate(hdf5_load_dataset(data_args.h5_data_file_path,
                                                                           all_indices,
                                                                           step_size,
                                                                           large_batch=data_args.large_batch,
                                                                           is_decode_utf8=is_decode_utf8,
                                                                           read_npy_from_disk=read_npy_from_disk)):
            train_dataset = (
                get_dataset_bpe(data_args,
                                tokenizer=tokenizer,
                                use_time_embed=model_args.use_time_embed,
                                debugN=model_args.debugN,
                                hdf_data=large_batch_data,
                                is_timestamp_rnn=data_args.is_timestamp_rnn,
                                embedding_cache_dir=embedding_cache_dir
                                ) if training_args.do_train else None
            )

            # Initialize our Trainer
            trainer = BpeTrainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                prediction_loss_only=True,
                loss_tracker=loss_tracker,
                optimizers=(optimizer, lr_scheduler)
            )

            trainer.args.output_dir = new_output_dir

            # Training
            if training_args.do_train:
                model_path = (
                    model_args.model_name_or_path
                    if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
                    else None
                )
                train_output = trainer.train(model_path=model_path,
                                             use_time_embed=model_args.use_time_embed,
                                             compute_total_flos=False,
                                             find_unused_parameters=find_unused_parameters,
                                             is_curriculum_within_task=data_args.is_curriculum_within_task,
                                             curr_learner=curr_learner,
                                             verbose=2,
                                             epoch_i=epoch_i,
                                             masker_recorder=masker_recorder,
                                             dynamic_mask_update_alpha=data_args.dynamic_mask_update_alpha,
                                             mask_type=data_args.mask_type,
                                             max_epoch=data_args.outer_epoch,
                                             mask_predict_prob_update_step=4,
                                             snapshot_gap=data_args.snapshot_gap,
                                             shuffle_epoch=data_args.shuffle_epoch,
                                             total_training_steps=total_training_steps
                                             )
                optimizer, lr_scheduler = trainer.optimizer, trainer.lr_scheduler

                if training_args.local_rank in {-1, 0}:

                    if large_batch_i == step_size - 1:

                        if epoch_i == data_args.outer_epoch - 1:
                            trainer.model.save_pretrained(trainer.args.output_dir)
                            logger.info(f"Saving model checkpoint to {os.path.abspath(training_args.output_dir)},"
                                        f" local rank: {training_args.local_rank}")
                        else:
                            if not masker_recorder.record_snapshot:
                                # Save for each epoch
                                temp_save_dir = os.path.join(trainer.args.output_dir,
                                                             f'outer_epoch_{epoch_i}_large_batch_{large_batch_i}')
                                trainer.model.save_pretrained(temp_save_dir)
                                logger.info(f"Saving model checkpoint to {temp_save_dir}")

                    logger.info(f"outer epoch-{epoch_i}/{data_args.outer_epoch - 1},"
                                f" large_batch-{large_batch_i}/{step_size - 1} training done,"
                                f" local_rank-{training_args.local_rank}"
                                f" data size: {large_batch_data.shape}")

        if masker_recorder is not None and data_args.is_record_snapshot:
            save_train_predict_prob(masker_recorder, tokenizer, new_output_dir, data_args)

    print(f"[Total Step] Total step: {total_training_steps}")
    save_meta_config(model_args, data_args, new_output_dir, data_collator, total_training_steps)

    # save predict y prob
    if masker_recorder is not None:
        save_train_predict_prob(masker_recorder, tokenizer, new_output_dir, data_args)

    # # Train adapter
    # if data_args.is_train_time_freq_adapter:
    #     from pretrain_adapter.train_adapter import prepare_adapater_data
    #     prepare_adapater_data(new_output_dir, data_args.h5_data_file_path, all_indices, step_size,
    #                           data_args.large_batch)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    total_time = end_time - start_time
    total_time_in_min = total_time / 60
    total_time_in_hour = total_time / 60 / 60
    print(f"[Total TIME] IN Min: {total_time_in_min} minutes, In hour: {total_time_in_hour} hours")
