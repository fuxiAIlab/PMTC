"""
References:
    - https://gitlab.leihuo.netease.com/userpersona/fumo/research/gamebert/-/tree/master/downstream_tasks_evaluation


"""
import os
import sys
import json
import ipdb
import copy
import torch
import ntpath
import subprocess
from tqdm import tqdm

sys.path.append('..')
sys.path.append('../..')

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from core.game_train import GameTrainerCallback
from core.config import train_config
from core.bot_detect_model import BotDetectModel
from core.buy_time_predict_model import BuyTimePredictModel
from core.union_recommend_predict_model import UnionRecommendModel
from core.downstream_dataset import DummyEmbedderWithoutPretrain
from core.churn_predict_rnn_model import ChurnPredictRnnModel
from core.bpe_bow_embedder import BpeBowEmbedder
from torch.utils.data import DataLoader
from game_seq_embedder import init_behavior_sequence_embedder
from core.downstream_task import BotDetectTask, ChurnPredictTask, ClusteringTask, BuyTimePredictTask, UnionRecommendTask
from core.downstream_dataset import create_collate_fn_bot_detect
from core.downstream_dataset import create_collate_fn_churn_predict
from core.downstream_dataset import create_collate_fn_buy_time_predict
from core.downstream_dataset import create_collate_fn_union_recommend
from core.captum_interpreter import CaptumInterpreter

import argparse

TASK_DICT = {
    'bot_detect': BotDetectTask,
    'churn_predict': ChurnPredictTask,
    'clustering': ClusteringTask,
    'buy_time_predict': BuyTimePredictTask,
    'union_recommend': UnionRecommendTask,
    'clustering_union': ClusteringTask
}

# Data paths:
# ../data/bot_detect_debug.h5
# ../data/churn_predict_debug.h5

"""

# DEBUG
python3.6 main.py --task_data_path data/bot_detect.h5 \
                  --task_name bot_detect \
                  --pretrain_models game_bert_NO_time_embed_whitespace game_bert_time_embed_sin_whitespace \
                  --compare_finetune 0 \
                  --debug_N 1000 \
                  --feature_choices '0 1 0' \
                  --is_debug 1 \
                  --test_n_split 3 \
                  --gpus 0

# Quick Debug
python3.6 main.py --task_data_path data/bot_detect_debug.h5 \
                  --task_name bot_detect \
                  --pretrain_models game_bert_NO_time_embed_BPE game_bert_NO_time_embed_whitespace \
                  --compare_finetune 0 \
                  --debug_N 200 \
                  --feature_choices '0 1 1' \
                  --is_debug 1 \
                  --test_n_split 3 \
                  --gpus 0

# Normal Debug
python3.6 main.py --task_data_path data/bot_detect.h5 \
                  --task_name bot_detect \
                  --pretrain_models game_bert_NO_time_embed_BPE game_bert_NO_time_embed_whitespace \
                  --compare_finetune 0 \
                  --debug_N 2000 \
                  --feature_choices '1 1 1' \
                  --is_debug 1 \
                  --test_n_split 5 \
                  --gpus 0
                  
python3.6 main.py --task_data_path data/bot_detect.h5 \
                  --task_name bot_detect \
                  --pretrain_models game_bert_time_embed_sin_whitespace_sep_des_id_t0_t1_t2_t3_t4_curr_task_step_20_alpha_2.0 \
                  --compare_finetune 0 \
                  --debug_N 500 \
                  --feature_choices '0 0 1' \
                  --is_debug 1 \
                  --test_n_split 5 \
                  --gpus 0

# Test random embedder
python3.6 main.py --task_data_path data/bot_detect.h5 \
                  --task_name bot_detect \
                  --pretrain_models random_embedder \
                  --compare_finetune 0 \
                  --debug_N 500 \
                  --feature_choices '0 0 1' \
                  --is_debug 1 \
                  --test_n_split 5 \
                  --gpus 0
                  
# Temp debug
python3.6 main.py --task_data_path data/bot_detect.h5 \
                  --task_name bot_detect \
                  --pretrain_models game_bert_time_embed_sin_whitespace_sep_des_id_time_gate \
                  --compare_finetune 0 \
                  --debug_N 2000 \
                  --feature_choices '0 1 1' \
                  --is_debug 1 \
                  --test_n_split 5 \
                  --gpus 0
"""

import logging

logging.getLogger("lightning").setLevel(logging.ERROR)


def load_save_json(json_path, mode, verbose=1, encoding='utf-8', data=None):
    if mode == 'save':
        assert data is not None
        with open(json_path, 'w', encoding=encoding) as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            if verbose >= 1:
                print(f"save json data to {json_path}")
    elif mode == 'load':
        if os.path.isfile(json_path):
            with open(json_path, 'r') as f:
                response = json.load(f)
            if verbose >= 1:
                print(f"load json from {json_path} success")
        else:
            raise Exception(f"{json_path} does not exist!")
        return response
    else:
        raise NotImplementedError


def cv_clustering_loop(target_task, repeat_percent=0.5, embedder_extract_batch_size=2):
    import copy
    import random
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from sklearn.decomposition import PCA

    input_seqs, cluster_labels = target_task.all_examples
    all_indices = list(range(len(input_seqs)))

    for cv_index in range(target_task.test_n_split):

        # Get Input embeddings
        random_indices = random.sample(all_indices, int(repeat_percent * len(input_seqs)))
        random_input_seqs = input_seqs[random_indices]
        random_input_seqs = random_input_seqs.astype(str)
        random_input_seqs = [[x for x in x_seq if x != '[PAD]'] for x_seq in random_input_seqs]
        input_embeddings = target_task.embedder.embed(random_input_seqs, batch_size=embedder_extract_batch_size,
                                                      layer=-2).detach().cpu().numpy()
        if input_embeddings.shape[1] > 768:
            pca = PCA(n_components=min(768, len(input_embeddings)))
            pca_input_embeddings = pca.fit_transform(input_embeddings)
            print(f"PCA done! Start Kmeans... Shape: {pca_input_embeddings.shape}")
        else:
            pca_input_embeddings = input_embeddings

        for (clustering_task, cluster_label) in cluster_labels:
            actual_labels = cluster_label[random_indices]
            n_clusters = len(set(actual_labels))
            predict_labels = KMeans(n_clusters=n_clusters, random_state=0, verbose=0).fit(pca_input_embeddings).labels_
            shuffle_labels = copy.deepcopy(actual_labels)
            random.shuffle(shuffle_labels)
            shuffle_ari_score = adjusted_rand_score(actual_labels, shuffle_labels)
            ari_score = adjusted_rand_score(actual_labels, predict_labels)
            target_task.log_train_result(cv_index,
                                         train_losses=None,
                                         val_losses=None,
                                         test_loss=None,
                                         test_metric=ari_score,
                                         test_metric_shuffle=None,
                                         train_size=len(random_indices),
                                         test_size=None,
                                         train_labels=n_clusters,
                                         train_date_ranges=None,
                                         test_labels=None,
                                         test_date_ranges=None,
                                         clustering_task=clustering_task
                                         )
            print(f"clustering_task: {clustering_task},"
                  f" Clustering ari score: {ari_score},"
                  f" shuffle_score: {shuffle_ari_score},"
                  f" n_clusters: {n_clusters}, "
                  f"train_size: {len(random_indices)}")
    target_task.save_log_result()
    return


def _temp_update_model_meta_config(pretrain_model_dir, embedder, embedder_meta_config):
    """
    临时性的，主要是为了获取模型预训练时的超参数
    Returns
    -------
    """
    # Temp, get softmax_t & alpha & warmup decay from path
    import re
    softmax_t = re.findall(r'_t-([0-9\.]+)', pretrain_model_dir)
    update_prob_alpha = re.findall(r'_alpha-([0-9\.]+)', pretrain_model_dir)
    warmup_decay = re.findall(r'_decay-([0-9\.]+)', pretrain_model_dir)
    if softmax_t:
        embedder_meta_config['softmax_t'] = softmax_t[0]
    if update_prob_alpha:
        embedder_meta_config['update_prob_alpha'] = update_prob_alpha[0]
    if warmup_decay:
        embedder_meta_config['warmup_decay'] = warmup_decay[0]
    embedder.meta_config = embedder_meta_config


def _get_finetune_params(embedder):
    if embedder.pretrain_task.endswith('adapter'):
        transformer_encoder_params = embedder.parameters
    else:
        transformer_encoder_params = embedder.model.parameters()
    return transformer_encoder_params


def _deep_copy_embedder(embedder):
    #         embed_tokenizer = Tokenizer.from_file(BPE_TOKENIZER_PATH)
    #         embed_tokenizer.cls_token_id = 0
    #         embed_tokenizer.pad_token_id = 1
    #         embed_tokenizer.sep_token_id = 2
    #         embed_tokenizer.unk_token_id = 3
    #         embed_tokenizer.mask_token_id = 4
    #         # tokenizer.get_special_tokens_mask = create_func1(tokenizer.pad_token_id, tokenizer.cls_token_id)
    #         # tokenizer.added_tokens_encoder = {}
    #         # tokenizer.convert_tokens_to_ids = create_func2(tokenizer.added_tokens_encoder, tokenizer.mask_token_id)
    #         # tokenizer.mask_token = '[MASK]'
    #         embed_tokenizer._pad_token = '[PAD]'
    #         embed_tokenizer.use_bpe = True
    #         embed_tokenizer.game_id_cn_char_map = load_save_json(BPE_GAME_ID_CN_CHAR_MAP_PATH, 'load', encoding='utf-8')
    #         embed_tokenizer.vocab = {'[PAD]': embed_tokenizer.pad_token_id}

    copy_embedder = copy.deepcopy(embedder)
    origin_tokenizer = embedder.tokenizer

    target_tokenizers = [copy_embedder.tokenizer]
    if hasattr(copy_embedder, 'transformer_embedder'):
        target_tokenizers.append(copy_embedder.transformer_embedder.tokenizer)

    for tokenizer in target_tokenizers:
        tokenizer.vocab = origin_tokenizer.vocab
        tokenizer.use_bpe = origin_tokenizer.use_bpe
        tokenizer.game_id_cn_char_map = origin_tokenizer.game_id_cn_char_map

    return copy_embedder


def cv_train_loop(target_task, task_name, epoch, batch_size, gpus, is_finetune,
                  accumulate_grad_batches=1,
                  interpreter=None):
    cv_tqdm = tqdm(target_task.train_val_test_datasets, total=target_task.test_n_split)

    with cv_tqdm:
        for cv_i, (cv_index, train_data, val_data, test_data) in enumerate(cv_tqdm):

            cv_tqdm.set_description(f"Task-{task_name}, cv_index: {cv_index}")

            # deep copy embedder
            if is_finetune:
                cv_embedder = _deep_copy_embedder(target_task.embedder)
                cv_transformer_encoder_params = _get_finetune_params(cv_embedder)
                if hasattr(target_task, 'union_embedder'):
                    cv_union_embedder = _deep_copy_embedder(target_task.union_embedder)
                    cv_union_transformer_encoder_params = _get_finetune_params(cv_union_embedder)
                else:
                    cv_union_embedder = None
                    cv_union_transformer_encoder_params = None

            else:
                cv_embedder = target_task.embedder
                cv_transformer_encoder_params = None
                cv_union_transformer_encoder_params = None

                if hasattr(target_task, 'union_embedder'):
                    if target_task.union_embedder is not None:
                        cv_union_embedder = target_task.union_embedder
                    else:
                        cv_union_embedder = None
                else:
                    cv_union_embedder = None

            # Train & set trainer
            trainer = pl.Trainer(
                max_epochs=epoch,
                deterministic=True,
                progress_bar_refresh_rate=5,
                num_sanity_val_steps=2,
                log_every_n_steps=50,
                gpus=gpus,
                accumulate_grad_batches=accumulate_grad_batches,
                logger=False,
                checkpoint_callback=False,
                callbacks=[GameTrainerCallback()]
            )
            trainer.is_finetune_bert = is_finetune
            trainer.embedder = cv_embedder
            trainer.interpreter = interpreter
            if interpreter is not None:
                trainer.target_task = target_task

            if cv_union_embedder is not None:
                trainer.union_embedder = target_task.union_embedder

            if task_name == 'bot_detect':
                if train_data.tokenizer is not None:
                    padding_idx = train_data.tokenizer.padding_idx
                    vocab_size = train_data.tokenizer.vocab_size
                else:
                    padding_idx = None
                    vocab_size = None
                task_model = BotDetectModel(
                    vocab_size=vocab_size,
                    padding_idx=padding_idx,
                    use_base_features=target_task.use_base_features,
                    use_pretrain_features=target_task.use_pretrain_features,
                    transformer_input_size=cv_embedder.embedding_dim,
                    transformer_encoder_params=cv_transformer_encoder_params)

                # set collate_fn
                if is_finetune:
                    collate_fn = create_collate_fn_bot_detect(embedder=cv_embedder)
                else:
                    collate_fn = create_collate_fn_bot_detect(embedder=None)

            elif task_name == 'churn_predict':
                task_model = ChurnPredictRnnModel(194,
                                                  rnn_hidden_size=256,
                                                  rnn_num_layer=1,
                                                  use_pretrain_features=target_task.use_pretrain_features,
                                                  transformer_input_size=cv_embedder.embedding_dim,
                                                  transformer_encoder_params=cv_transformer_encoder_params
                                                  )
                # task_model = CPTCN(14, 194,
                #                    use_pretrain_features=target_task.use_pretrain_features,
                #                    transformer_dim=target_task.embedder.embedding_dim)
                if is_finetune:
                    collate_fn = create_collate_fn_churn_predict(embedder=cv_embedder, batch_size=1)
                else:
                    collate_fn = None

            elif task_name == 'buy_time_predict':
                task_model = BuyTimePredictModel(transformer_input_size=cv_embedder.embedding_dim,
                                                 transformer_encoder_params=cv_transformer_encoder_params,
                                                 is_finetune=is_finetune)
                if is_finetune:
                    collate_fn = create_collate_fn_buy_time_predict(embedder=cv_embedder,
                                                                    batch_size=2,
                                                                    player_status_dict=target_task.player_status_dict)
                else:
                    collate_fn = None

            elif task_name == 'union_recommend':
                task_model = UnionRecommendModel(transformer_input_size=cv_embedder.embedding_dim,
                                                 transformer_encoder_params=cv_transformer_encoder_params,
                                                 union_transformer_encoder_params=cv_union_transformer_encoder_params,
                                                 is_finetune=is_finetune)
                collate_fn = create_collate_fn_union_recommend(cv_embedder,
                                                               union_embedder=cv_union_embedder,
                                                               player_status_dict=target_task.player_status_dict)
            else:
                raise NotImplementedError

            # Check Train/Test samples no over-lapping
            # 特别要注意，如果下面的x是torch.tensor，每次重新运行脚本都可能获得不同的hash值，原因未知
            train_val_data_hashes = [hash(x) for x in train_data.examples] + [hash(x) for x in val_data.examples]
            test_data_hashes = [hash(x) for x in test_data.examples]
            assert not set(train_val_data_hashes).intersection(test_data_hashes)

            seed_everything(cv_i)
            trainer.fit(task_model,
                        DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn),
                        DataLoader(val_data, batch_size=batch_size, collate_fn=collate_fn))

            # # ----------------------------------------------------------------------------------------------------------
            # # examine the activation of bert embedding layer
            # # ----------------------------------------------------------------------------------------------------------
            # for name, param in trainer.embedder.model.named_parameters():
            #
            #     # param shape: 50000 x 768
            #     if name == 'bert.embeddings.word_embeddings.weight':
            #         save_path = os.path.abspath(os.path.join('analyse_bow_bpe_input', f'finetune_{task_name}.csv'))
            #         KEEP_TOP = 100
            #         vocab = target_task.embedder.tokenizer.get_vocab()
            #         re_vocab = {y: x for x, y in vocab.items()}
            #         game_id_cn_char_dict = load_save_json('../../static/game_id_cn_char.dict', 'load')
            #         re_game_id_cn_char_dict = {y: x for x, y in game_id_cn_char_dict.items()}
            #         import numpy as np
            #         bpe_segment_value_save_path = os.path.abspath(os.path.join('analyse_bow_bpe_input',
            #                                                                    f'bpe_segment_avg_values.npy'))
            #         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            #         bpe_segment_avg_values = torch.tensor(np.load(bpe_segment_value_save_path)).to(device)
            #
            #         # train_data.tokenizer
            #         bpe_segment_avg_values_mask = bpe_segment_avg_values > 0.25
            #         sum_values, sum_index = torch.sort(torch.sum(torch.abs(param), dim=1), descending=True)
            #         sum_values_mask, sum_index_mask = sum_values[bpe_segment_avg_values_mask], sum_index[
            #             bpe_segment_avg_values_mask]
            #         bpe_segment_avg_values = bpe_segment_avg_values[bpe_segment_avg_values_mask]
            #         sum_values_mask = sum_values_mask / bpe_segment_avg_values
            #         sum_values_index = list(zip(sum_values_mask.tolist(), sum_index_mask.tolist()))
            #         sorted_sum_values_index = sorted(sum_values_index, key=lambda x: x[0], reverse=True)[:KEEP_TOP]
            #         sum_values, sum_index = [x[0] for x in sorted_sum_values_index], [x[1] for x in
            #                                                                           sorted_sum_values_index]
            #
            #         top_n_bpe_segments = [re_vocab[i] for i in sum_index]
            #         df = {'segments': [], 'sum': []}
            #         for sum_value, segment in zip(sum_values, top_n_bpe_segments):
            #             id_segment = [re_game_id_cn_char_dict.get(x, '') for x in segment]
            #             df['segments'].append(' '.join(id_segment))
            #             df['sum'].append(sum_value)
            #         df = pd.DataFrame(df)
            #         df.to_csv(save_path, index=False)
            #         print(f"Save df to {save_path}")
            #         sys.exit()
            # # ----------------------------------------------------------------------------------------------------------

            # # ----------------------------------------------------------------------------------------------------------
            # # examine the activation of bow inputs
            # # ----------------------------------------------------------------------------------------------------------
            # for name, param in task_model.named_parameters():
            #     if name == 'input_projection.weight':
            #
            #         save_path = os.path.abspath(os.path.join('analyse_bow_bpe_input', f'{task_name}.csv'))
            #         numpy_save_path =  os.path.abspath(os.path.join('analyse_bow_bpe_input', f'bpe_segment_avg_values.npy'))
            #
            #         KEEP_TOP = 100
            #         vocab = target_task.embedder.tokenizer.get_vocab()
            #         re_vocab = {y: x for x, y in vocab.items()}
            #         game_id_cn_char_dict = load_save_json('../../static/game_id_cn_char.dict', 'load')
            #         re_game_id_cn_char_dict = {y: x for x, y in game_id_cn_char_dict.items()}
            #
            #         # train_data.tokenizer
            #         bpe_segment_avg_values = train_data.bpe_segment_avg_values
            #         bpe_segment_avg_values_mask = bpe_segment_avg_values > 0.25
            #         sum_values, sum_index = torch.sort(torch.sum(torch.abs(param.t()), dim=1), descending=True)
            #         sum_values_mask, sum_index_mask = sum_values[bpe_segment_avg_values_mask], sum_index[
            #             bpe_segment_avg_values_mask]
            #         bpe_segment_avg_values = bpe_segment_avg_values[bpe_segment_avg_values_mask]
            #         sum_values_mask = sum_values_mask / bpe_segment_avg_values
            #         sum_values_index = list(zip(sum_values_mask.tolist(), sum_index_mask.tolist()))
            #         sorted_sum_values_index = sorted(sum_values_index, key=lambda x: x[0], reverse=True)[:KEEP_TOP]
            #         sum_values, sum_index = [x[0] for x in sorted_sum_values_index], [x[1] for x in
            #                                                                           sorted_sum_values_index]
            #
            #         top_n_bpe_segments = [re_vocab[i] for i in sum_index]
            #         df = {'segments': [], 'sum': []}
            #         for sum_value, segment in zip(sum_values, top_n_bpe_segments):
            #             id_segment = [re_game_id_cn_char_dict.get(x, '') for x in segment]
            #             df['segments'].append(' '.join(id_segment))
            #             df['sum'].append(sum_value)
            #         df = pd.DataFrame(df)
            #         df.to_csv(save_path, index=False)
            #         print(f"Save df to {save_path}")
            #         sys.exit()
            # # ----------------------------------------------------------------------------------------------------------

            trainer.test(task_model,
                         DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn))
            target_task.log_train_result(cv_index,
                                         train_losses=trainer.train_losses,
                                         val_losses=trainer.val_losses,
                                         test_loss=trainer.test_loss,
                                         test_metric=trainer.test_metric_value,
                                         test_metric_shuffle=trainer.test_metric_value_shuffle,
                                         train_size=train_data.size,
                                         test_size=test_data.size,
                                         train_labels=train_data.labels,
                                         train_date_ranges=train_data.date_ranges,
                                         test_labels=test_data.labels,
                                         test_date_ranges=test_data.date_ranges,
                                         )
    # Save task result for one model
    target_task.save_log_result()


class RandomEmbedder:

    def __init__(self):
        self.model_name = 'random'
        self.pretrain_task = 'random'
        self.embedding_cache = {}
        self.meta_config = {}
        self.max_sequence_length = 512
        self.embedding_dim = 768
        self.use_token_weights = False

    def embed(self, samples, **kwargs):
        return torch.rand((len(samples), self.embedding_dim))

    def update_embedding_cache(self, x1, x2):
        pass

    def save_embedding_cache(self):
        pass


def args_parse():
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('--task_name', type=str, choices=['bot_detect',
                                                          'churn_predict',
                                                          'clustering',
                                                          'buy_time_predict',
                                                          'union_recommend',
                                                          'clustering_union'], required=True)
    # parser.add_argument('--pretrain_base_dir', type=str, required=True)
    parser.add_argument('--log_base_dir', type=str, default='results')
    parser.add_argument('--log_save_dir', type=str)
    parser.add_argument('--pretrain_models', type=str, nargs='+')
    parser.add_argument('--task_data_path', type=str, required=True)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_n_split', type=int, default=10)
    parser.add_argument('--compare_finetune', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--is_debug', type=int, default=0)
    parser.add_argument('--debug_N', type=int, default=0)
    parser.add_argument('--gpus', type=int, nargs='+', default=None)
    parser.add_argument('--conca_output_tasks', type=str, nargs='+', default=None)
    parser.add_argument('--mask_prob', type=float, default=None)
    parser.add_argument('--mask_multiple_time', type=int, default=None)
    parser.add_argument('--is_mask_output_concat', type=int, default=0)
    parser.add_argument('--is_finetune', type=int, default=0)
    parser.add_argument('--output_mask_embedding', type=int, default=1)
    parser.add_argument('--save_embedding_cache', type=int, default=1)
    parser.add_argument('--add_player_status', type=int, default=0)
    parser.add_argument('--pretrain_base_dir', type=str)
    parser.add_argument('--do_interpretability', type=int)
    parser.add_argument('--feature_choices', type=str, default='1 0 0',
                        help='use only base features'
                             ' | use base features with pretrain features | use pretrain features only')
    args = parser.parse_args()
    return args


def main():
    args = args_parse()

    # Settings based on machine location

    host_id = subprocess.check_output('hostid').strip().decode('utf-8')
    pretrain_base_dir = '../../bert_model'
    embedding_cache_dir = None

    # if host_id == '007f0101':
    #     # set embedding cache path
    #     embedding_cache_dir = '/media/iamlxb3/2D97AD940A9AD661/temp_embedding_cache'
    #     if args.pretrain_base_dir is None:
    #         pretrain_base_dir = '/media/iamlxb3/2D97AD940A9AD661/model_ckpts/game_bert/'
    #     else:
    #         pretrain_base_dir = args.pretrain_base_dir
    # else:
    #     embedding_cache_dir = '/data/game_bert/temp_embedding_cache'
    #     if args.pretrain_base_dir is None:
    #         pretrain_base_dir = '/root/game_bert/bert_model/'
    #     else:
    #         pretrain_base_dir = args.pretrain_base_dir

    print(f"host_id: {host_id}, embedding_cache_dir: {embedding_cache_dir}")

    # if args.pretrain_models == ['0']:
    #     print("No pretrain_models argument passed")
    #     pretrain_models = os.listdir(args.pretrain_base_dir)
    #     print(f"-" * 78)
    #     print(f"Load all pretrained model from {args.pretrain_base_dir}. Total: {len(pretrain_models)}")
    #     print(f"-" * 78)
    #     for model_name in pretrain_models:
    #         print(model_name)
    # else:
    #     pretrain_models = args.pretrain_models

    # set random seed
    seed_everything(args.seed)

    # Read arguments
    val_ratio = args.val_ratio
    test_n_split = args.test_n_split
    log_base_dir = os.path.abspath(args.log_base_dir)
    task_name = args.task_name
    debug_N = args.debug_N
    is_debug = args.is_debug
    is_debug = True if is_debug else False
    pretrain_models = args.pretrain_models
    conca_output_tasks = args.conca_output_tasks
    mask_prob = args.mask_prob
    log_save_dir = args.log_save_dir
    mask_multiple_time = args.mask_multiple_time
    is_mask_output_concat = args.is_mask_output_concat
    output_mask_embedding = args.output_mask_embedding
    save_embedding_cache = args.save_embedding_cache
    is_finetune = args.is_finetune
    do_interpretability = True if args.do_interpretability else False
    is_finetune = True if is_finetune else False
    save_embedding_cache = True if save_embedding_cache else False
    is_mask_output_concat = True if is_mask_output_concat else False
    output_mask_embedding = True if output_mask_embedding else False

    # read player status data
    if args.add_player_status:
        import compress_pickle
        player_status_data_path = '../../data/player_status_info/player_status_feature.gz'
        assert os.path.isfile(player_status_data_path)
        player_status_dict = compress_pickle.load(player_status_data_path)
        print(f"Load player status dict from {player_status_data_path}, total: {len(player_status_dict)}")
    else:
        player_status_dict = None

    # if debug_N:
    #     assert is_debug

    gpus = args.gpus
    task_data_path = args.task_data_path
    feature_choices = args.feature_choices
    feature_choices = feature_choices.split(' ')
    feature_choices = [int(x) for x in feature_choices]
    use_base_features, use_both_features, use_pretrain_features = feature_choices

    # assert use_base_features or use_both_features or use_pretrain_features
    print(f"Gpus: {gpus}")

    # init
    if do_interpretability:
        interpreter = CaptumInterpreter(log_base_dir, task_name)
    else:
        interpreter = None

    # read configs
    if is_debug:
        epoch = train_config.epoch_debug
    else:
        if is_finetune:
            epoch = train_config.finetune_epoch
        else:
            epoch = train_config.epoch

    # TODO, temp
    if task_name == 'union_recommend' and not is_debug:
        epoch = train_config.union_recommend_epoch
        print(f"[Union Recommend] Set training epoch to {epoch}")
    print(f"Set training epoch to {epoch}")

    if is_finetune:
        batch_size = train_config.finetune_batch_size[task_name]
        original_batch_size = train_config.batch_size
        accumulate_grad_batches = original_batch_size // batch_size
        print(f"Task: {task_name}, Finetune batch size is set to {batch_size}, "
              f"accumulate_grad_batches: {accumulate_grad_batches}")
        assert accumulate_grad_batches >= 1
        assert 'bpe_bow' not in pretrain_models
    else:
        batch_size = train_config.batch_size
        accumulate_grad_batches = 1

    # TODO, Load Data First
    task_all_examples = None  # 这里这样设定的目的就是不要多次读取数据了，数据读一次分割一次就够了

    if task_name in {'clustering', 'buy_time_predict', 'union_recommend'}:
        assert not use_both_features
        assert not use_base_features
        assert use_pretrain_features
    if task_name == 'union_recommend':
        assert len(pretrain_models) == 1

    if use_base_features:
        # Train without Pretrain feature
        embedder_without_pretrain = DummyEmbedderWithoutPretrain()
        target_task = TASK_DICT[task_name](embedder_without_pretrain,
                                           task_data_path=task_data_path,
                                           test_n_split=test_n_split,
                                           val_ratio=val_ratio,
                                           task_name=task_name,
                                           log_base_dir=log_base_dir,
                                           log_save_dir=log_save_dir,
                                           debug_N=debug_N,
                                           is_debug=is_debug,
                                           use_base_features=True,
                                           use_pretrain_features=False,
                                           all_examples=task_all_examples,
                                           is_finetune=is_finetune,
                                           player_status_dict=player_status_dict,
                                           )
        if target_task.is_result_file_exist:
            print(f"Find result log exist, skip training! save_path: {target_task.result_log_save_path}")
        else:
            task_all_examples = target_task.all_examples
            cv_train_loop(target_task, task_name, epoch, batch_size, gpus,
                          is_finetune=is_finetune,
                          accumulate_grad_batches=accumulate_grad_batches,
                          interpreter=interpreter)

    if use_pretrain_features or use_both_features:
        pretrain_model_dirs = [os.path.join(pretrain_base_dir, x) for x in pretrain_models]
        for pretrain_model_dir in pretrain_model_dirs:

            # if pretrain_model_dir.endswith('adapter'):
            #     batch_size_adjust_ratio = 2
            #     batch_size = int(batch_size / batch_size_adjust_ratio)
            #     accumulate_grad_batches = accumulate_grad_batches * batch_size_adjust_ratio
            #     print(
            #         f"Adjust model-{ntpath.basename(pretrain_model_dir)} batch_size to {batch_size},"
            #         f" accumulate_grad_batches to {accumulate_grad_batches}")

            meta_config_path = os.path.join(pretrain_model_dir, 'meta_config.json')
            if os.path.isfile(meta_config_path):
                meta_config = load_save_json(meta_config_path, 'load')
                is_train_adapter = meta_config.get('is_train_adapter', False)
            else:
                is_train_adapter = False

            # TODO, ADD FINETUNE
            if 'random_embedder' in pretrain_model_dir:
                embedder = RandomEmbedder()
            elif 'bpe_bow' in pretrain_model_dir:
                embedder = BpeBowEmbedder(bpe_tokenizer_path='../../static/bpe_new.str',
                                          cn_char_id_mapping_path='../../static/game_id_cn_char.dict')
                if host_id == '007f0101':
                    batch_size = 8
            else:
                # Init Embedder
                embedder = init_behavior_sequence_embedder(pretrain_model_dir,
                                                           is_finetune=is_finetune,
                                                           embedding_cache_dir=embedding_cache_dir)
                embedder.pretrain_task = ntpath.basename(pretrain_model_dir).replace('game_bert_', '')
                embedder.load_embedding_cache()
                meta_config_path = os.path.join(pretrain_model_dir, 'meta_config.json')
                embedder_meta_config = load_save_json(meta_config_path, 'load')

                # Temp, get softmax_t & alpha & warmup decay from path
                _temp_update_model_meta_config(pretrain_model_dir, embedder, embedder_meta_config)

            if player_status_dict is not None:
                if task_name in {'buy_time_predict', 'union_recommend'}:
                    embedder.embedding_dim += 73
                else:
                    raise NotImplementedError

            # special for clustering
            if task_name.startswith('clustering'):
                target_task = TASK_DICT[task_name](embedder,
                                                   task_data_path=task_data_path,
                                                   test_n_split=test_n_split,
                                                   val_ratio=val_ratio,
                                                   task_name=task_name,
                                                   log_base_dir=log_base_dir,
                                                   log_save_dir=log_save_dir,
                                                   debug_N=debug_N,
                                                   is_debug=is_debug,
                                                   use_base_features=False,
                                                   use_pretrain_features=True,
                                                   all_examples=task_all_examples,
                                                   save_embedding_cache=save_embedding_cache,
                                                   is_finetune=is_finetune,
                                                   player_status_dict=player_status_dict
                                                   )
                if target_task.is_result_file_exist:
                    print(f"Find result log exist, skip training! save_path: {target_task.result_log_save_path}")
                else:
                    task_all_examples = target_task.all_examples
                    if host_id == '007f0101':
                        embedder_extract_batch_size = 2
                    else:
                        embedder_extract_batch_size = 8
                    cv_clustering_loop(target_task, embedder_extract_batch_size=embedder_extract_batch_size)
                continue
            # Train by only pretrain features
            if use_pretrain_features:
                # Init Task Object
                target_task = TASK_DICT[task_name](embedder,
                                                   task_data_path=task_data_path,
                                                   test_n_split=test_n_split,
                                                   val_ratio=val_ratio,
                                                   task_name=task_name,
                                                   log_base_dir=log_base_dir,
                                                   log_save_dir=log_save_dir,
                                                   debug_N=debug_N,
                                                   is_debug=is_debug,
                                                   use_base_features=False,
                                                   use_pretrain_features=True,
                                                   all_examples=task_all_examples,
                                                   save_embedding_cache=save_embedding_cache,
                                                   is_finetune=is_finetune,
                                                   player_status_dict=player_status_dict,
                                                   do_interpretability=do_interpretability
                                                   )
                if do_interpretability:
                    target_task.save_file_name = f'{task_name}_debugN-{debug_N}_interpretability.csv'
                    target_task.result_log_save_path = os.path.join(target_task.log_save_dir,
                                                                    target_task.save_file_name)

                if target_task.is_result_file_exist:
                    print(f"Find result log exist, skip training! save_path: {target_task.result_log_save_path}")
                else:
                    task_all_examples = target_task.all_examples
                    cv_train_loop(target_task, task_name, epoch, batch_size, gpus,
                                  is_finetune=is_finetune,
                                  accumulate_grad_batches=accumulate_grad_batches,
                                  interpreter=interpreter)
                    # Save interpreter data
                    if interpreter is not None:
                        interpreter.save_all_to_disk()

            # Train by both features
            if use_both_features:
                # Init Task Object
                target_task = TASK_DICT[task_name](embedder,
                                                   task_data_path=task_data_path,
                                                   test_n_split=test_n_split,
                                                   val_ratio=val_ratio,
                                                   task_name=task_name,
                                                   log_base_dir=log_base_dir,
                                                   log_save_dir=log_save_dir,
                                                   debug_N=debug_N,
                                                   is_debug=is_debug,
                                                   use_base_features=True,
                                                   use_pretrain_features=True,
                                                   all_examples=task_all_examples,
                                                   save_embedding_cache=save_embedding_cache,
                                                   is_finetune=is_finetune,
                                                   player_status_dict=player_status_dict
                                                   )
                if target_task.is_result_file_exist:
                    print(f"Find result log exist, skip training! save_path: {target_task.result_log_save_path}")
                else:
                    task_all_examples = target_task.all_examples
                    cv_train_loop(target_task, task_name, epoch, batch_size, gpus,
                                  is_finetune=is_finetune,
                                  accumulate_grad_batches=accumulate_grad_batches,
                                  interpreter=interpreter)


if __name__ == '__main__':
    main()
