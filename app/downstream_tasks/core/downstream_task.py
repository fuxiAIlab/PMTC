import os
import shutil
import pandas as pd
import numpy as np
import ipdb
import math
import glob
import copy
import torch
import compress_pickle
import pickle
import random
import ntpath
import hashlib
import collections
import h5py

from tqdm import tqdm
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from .downstream_dataset import BotDetectDataset
from .downstream_dataset import BotDetectDatasetFinetune
from .downstream_dataset import ChurnPredictDataset
from .downstream_dataset import ChurnPredictDatasetFinetune
from .downstream_dataset import BuyTimePredictDataset
from .downstream_dataset import BuyTimePredictFinetuneDataset
from .downstream_dataset import UnionRecommendDataset
from .downstream_tokenzier import WhitespaceTokenizer
from .config import train_config
from .utils import load_save_json


class GameTask:
    def __init__(self,
                 embedder,
                 test_n_split,
                 val_ratio=0.1,
                 task_name=None,
                 log_base_dir='',
                 log_save_dir=None,
                 overwrite=False,
                 debug_N=None,
                 is_debug=False,
                 use_base_features=False,
                 use_pretrain_features=False,
                 save_embedding_cache=True,
                 is_finetune=False,
                 pos_balance_ratio=None,
                 player_status_dict=None,
                 do_interpretability=False,
                 ):
        self.embedder = embedder
        self._train_val_test_datasets = []
        self.test_n_split = test_n_split
        self.val_ratio = val_ratio
        self.task_name = task_name
        self.type = type
        self.debug_N = debug_N
        self.is_debug = is_debug
        self.use_base_features = use_base_features
        self.use_pretrain_features = use_pretrain_features
        self.is_finetune = is_finetune
        self.pos_balance_ratio = pos_balance_ratio
        self.player_status_dict = player_status_dict
        self.do_interpretability = do_interpretability

        if log_save_dir is not None:
            self.log_save_dir = os.path.abspath(log_save_dir)
        else:
            self.log_save_dir = os.path.join(log_base_dir, task_name)

        self.save_embedding_cache = save_embedding_cache
        self.log_dict = {
            'cv_index': [],
            'train_losses': [],
            'val_losses': [],
            'test_loss': [],
            'test_metric': [],
            'test_metric_shuffle': [],
            'train_size': [],
            'test_size': [],
            'train_labels': [],
            'test_labels': [],
            'train_date_ranges': [],
            'test_date_ranges': [],
            'clustering_task': []
        }

        if overwrite:
            if os.path.isdir(self.log_save_dir):
                shutil.rmtree(self.log_save_dir)
                print(f"[Game Task] Overwrite is set to True, remove dir {self.log_save_dir}")

        if not os.path.isdir(self.log_save_dir):
            os.makedirs(self.log_save_dir)
            print(f"[Game Task] Make new dir {self.log_save_dir} for task-{self.task_name}")

        self.save_file_name = self._get_save_file_name()
        self.result_log_save_path = os.path.join(self.log_save_dir, self.save_file_name)

    def _get_save_file_name(self):
        if self.do_interpretability:
            save_file_path = f'{self.task_name}_debugN-{self.debug_N}_interpretability.csv'
        else:
            save_file_path = f'{self.embedder.model_name}_{self.embedder.pretrain_task}' \
                             f'_token_weights-{self.embedder.use_token_weights}' \
                             f'_{self.use_base_features}_{self.use_pretrain_features}' \
                             f'_split-{self.test_n_split}_finetune-{self.is_finetune}_debug-{self.debug_N}' \
                             f'_player_state-{True if self.player_status_dict is not None else False}.csv'

        return save_file_path

    @property
    def is_result_file_exist(self):
        if os.path.isfile(self.result_log_save_path):
            print(f"Find existing file name: {self.result_log_save_path}")
            return True
        else:
            print(f"Save file path-{self.result_log_save_path} not Found!")
            return False

    def load_existing_train_val_test_datasets(self, dataset):
        self._train_val_test_datasets = dataset

    @property
    def train_val_test_datasets(self):
        assert self._train_val_test_datasets
        for cv_index, (train_data, val_data, test_data) in enumerate(self._train_val_test_datasets):
            yield cv_index, train_data, val_data, test_data

    def log_train_result(self,
                         cv_index,
                         train_losses,
                         val_losses,
                         test_loss,
                         test_metric,
                         train_size,
                         test_size,
                         train_labels,
                         test_labels,
                         train_date_ranges,
                         test_date_ranges,
                         test_metric_shuffle,
                         clustering_task=None
                         ):
        self.log_dict['cv_index'].append(cv_index)
        self.log_dict['train_losses'].append(train_losses)
        self.log_dict['val_losses'].append(val_losses)
        self.log_dict['test_loss'].append(test_loss)
        self.log_dict['test_metric'].append(test_metric)
        self.log_dict['test_metric_shuffle'].append(test_metric_shuffle)
        self.log_dict['train_size'].append(train_size)
        self.log_dict['test_size'].append(test_size)
        self.log_dict['train_labels'].append(train_labels)
        self.log_dict['test_labels'].append(test_labels)
        self.log_dict['train_date_ranges'].append(train_date_ranges)
        self.log_dict['test_date_ranges'].append(test_date_ranges)
        self.log_dict['clustering_task'].append(clustering_task)

    def save_log_result(self):
        df = pd.DataFrame(self.log_dict)
        df['task'] = self.task_name
        df['use_base_features'] = self.use_base_features
        df['use_pretrain_features'] = self.use_pretrain_features
        df['model'] = self.embedder.model_name
        pretrain_task = self.embedder.pretrain_task + f'_{self.embedder.use_token_weights}'
        if self.player_status_dict is not None:
            pretrain_task += '_with_player_status'

        df['pretrain_task'] = pretrain_task
        df['debug_N'] = self.debug_N
        df['is_debug'] = self.is_debug
        df['embedder_max_len'] = self.embedder.max_sequence_length
        df['use_token_weights'] = self.embedder.use_token_weights
        df['is_finetune'] = self.is_finetune

        if hasattr(self.embedder, 'meta_config'):
            update_meta_keys = ['softmax_t', 'update_prob_alpha', 'warmup_decay']
            for meta_key in update_meta_keys:
                if meta_key in self.embedder.meta_config:
                    df[f'embedder_{meta_key}'] = self.embedder.meta_config[meta_key]

        df.to_csv(self.result_log_save_path, index=False)
        print(f"Save log file to {self.result_log_save_path}, shape: {df.shape}")

    def _split_train_val(self, train_val_examples):
        if self.task_name in {'bot_detect', 'churn_predict'}:
            train_val_labels = [x[1] for x in train_val_examples]
            train_examples, val_examples = train_test_split(train_val_examples,
                                                            test_size=self.val_ratio,
                                                            random_state=1,
                                                            stratify=train_val_labels)
        elif self.task_name == 'map_preload':
            # train_val_labels = [x[-1] for x in train_val_examples] # Used for stratify split
            train_examples, val_examples = train_test_split(train_val_examples,
                                                            test_size=self.val_ratio,
                                                            random_state=1)
        else:
            raise Exception

        return train_examples, val_examples

    def _get_role_id_with_complete_label(self, examples, src_role_ids, sample_n, max_try=20):
        try_time = 0
        valid_role_ids = None
        while try_time <= max_try:
            try_time += 1
            temp_role_ids = set(random.sample(src_role_ids, sample_n))
            label_set = set([x[-1] for x in examples if x[0] in temp_role_ids])
            if len(label_set) > 1:
                valid_role_ids = temp_role_ids
                break
        return valid_role_ids

    def _split_cv_by_role_id(self, all_examples, val_ratio=0.1):
        all_role_ids = [x[0] for x in all_examples]
        unique_role_ids = set(all_role_ids)
        test_set_size = int(len(unique_role_ids) / self.test_n_split)

        for i in range(self.test_n_split):

            # get valid test role ids
            test_role_ids = self._get_role_id_with_complete_label(all_examples, unique_role_ids, test_set_size)
            if test_role_ids is None:
                raise Exception("Couldn't find test set with complete labels")
            else:
                train_val_role_ids = [x for x in unique_role_ids if x not in test_role_ids]

            # create small training dataset according to debugN
            if not self.is_debug:
                assert self.debug_N
                debug_N_ratio = self.debug_N / len(all_examples)
                debug_train_sample_size = min(int(debug_N_ratio * len(train_val_role_ids)),
                                              len(train_val_role_ids))
                try_n = 0
                max_try = 20
                while True:
                    # make sure train/val role ids all valid
                    train_val_role_ids = self._get_role_id_with_complete_label(all_examples,
                                                                               train_val_role_ids,
                                                                               debug_train_sample_size)
                    val_n = int(len(train_val_role_ids) * val_ratio)
                    val_n = max(1, val_n)
                    val_role_ids = self._get_role_id_with_complete_label(all_examples, train_val_role_ids, val_n)
                    val_label_set = set([x[-1] for x in all_examples if x[0] in val_role_ids])
                    train_role_ids = set([x for x in train_val_role_ids if x not in val_role_ids])
                    train_label_set = set([x[-1] for x in all_examples if x[0] in train_role_ids])
                    if len(val_label_set) > 1 and len(train_label_set) > 1:
                        break
                    try_n += 1
                    if try_n >= max_try:
                        raise Exception("Can't find valid train/val at the same time")

                print(f"[Sample debug training data] train/val unique role id size: {len(train_val_role_ids)},"
                      f" debug_N_ratio: {debug_N_ratio},"
                      f" debug_N: {self.debug_N}, total size: {len(all_examples)}")
            else:
                # get valid val role ids
                val_n = int(len(train_val_role_ids) * val_ratio)
                val_n = max(1, val_n)
                val_role_ids = self._get_role_id_with_complete_label(all_examples, train_val_role_ids, val_n)
                train_role_ids = set([x for x in train_val_role_ids if x not in val_role_ids])
                if val_role_ids is None:
                    raise Exception("Couldn't find valiation set with complete labels")

            assert set(train_val_role_ids).intersection(test_role_ids) == set()
            assert set(val_role_ids).intersection(train_role_ids) == set()
            train_data = [x for x in all_examples if x[0] in train_role_ids]

            # Up sampling for training data
            if self.pos_balance_ratio is not None:
                pos_indices = [i for i, x in enumerate(train_data) if x[2] == 1]
                neg_indices = [i for i, x in enumerate(train_data) if x[2] == 0]

                to_sample_n = int(len(neg_indices) * self.pos_balance_ratio - len(pos_indices))
                sampled_pos_indices = [random.choice(pos_indices) for _ in range(to_sample_n)]
                train_data = np.array(train_data)
                sampled_pos_data = train_data[sampled_pos_indices]
                train_data = list(train_data) + list(sampled_pos_data)

                after_pos_indices = [i for i, x in enumerate(train_data) if x[2] == 1]
                print(
                    f"[Balance training data] after pos/neg sampling ratio: {len(after_pos_indices) / len(neg_indices)}"
                    f", size: {len(train_data)}")

            val_data = [x for x in all_examples if x[0] in val_role_ids]
            test_data = [x for x in all_examples if x[0] in test_role_ids]

            if self.task_name == 'buy_time_predict':
                if self.is_finetune:
                    train_dataset = BuyTimePredictFinetuneDataset(train_data,
                                                                  player_status_dict=self.player_status_dict)
                    val_dataset = BuyTimePredictFinetuneDataset(val_data,
                                                                player_status_dict=self.player_status_dict)
                    test_dataset = BuyTimePredictFinetuneDataset(test_data,
                                                                 player_status_dict=self.player_status_dict)
                else:
                    train_dataset = BuyTimePredictDataset(train_data, embedder=self.embedder,
                                                          player_status_dict=self.player_status_dict)
                    val_dataset = BuyTimePredictDataset(val_data, embedder=self.embedder,
                                                        player_status_dict=self.player_status_dict)
                    test_dataset = BuyTimePredictDataset(test_data, embedder=self.embedder,
                                                         player_status_dict=self.player_status_dict)
                self._train_val_test_datasets.append((train_dataset, val_dataset, test_dataset))
                print(f"[DataSet Create for {self.task_name}], "
                      f"train_size: {len(train_data)},"
                      f" val_size: {len(val_data)},"
                      f" test_size: {len(test_data)},"
                      f" train_label: {collections.Counter([x[2] for x in train_data])},"
                      f" val_label: {collections.Counter([x[2] for x in val_data])},"
                      f" test_label: {collections.Counter([x[2] for x in test_data])}")
            elif self.task_name == 'union_recommend':
                train_dataset = UnionRecommendDataset(train_data)
                val_dataset = UnionRecommendDataset(val_data)
                test_dataset = UnionRecommendDataset(test_data)
                self._train_val_test_datasets.append((train_dataset, val_dataset, test_dataset))
                print(f"[DataSet Create for {self.task_name}], "
                      f"train_size: {len(train_data)},"
                      f" val_size: {len(val_data)},"
                      f" test_size: {len(test_data)},"
                      f" train_label: {collections.Counter([x[4] for x in train_data])},"
                      f" val_label: {collections.Counter([x[4] for x in val_data])},"
                      f" test_label: {collections.Counter([x[4] for x in test_data])}")
            else:
                raise NotImplementedError

    def split_cv(self, all_examples):
        # kf = KFold(n_splits=self.test_n_split, shuffle=True)
        kf = StratifiedKFold(n_splits=self.test_n_split, random_state=1, shuffle=True)
        labels = [x[1] for x in all_examples]
        labels = np.array(labels)
        for split_i, (train_val_index, test_index) in enumerate(kf.split(all_examples, labels)):
            random.shuffle(train_val_index)
            train_ratio = float(self.debug_N / len(all_examples))
            train_end_index = int(train_ratio * len(train_val_index))
            train_val_index_part = train_val_index[:train_end_index]
            train_val_examples_part = all_examples[train_val_index_part]
            test_examples = all_examples[test_index]
            train_examples, val_examples = self._split_train_val(train_val_examples_part)

            train_labels = [x[1] for x in train_examples]
            val_labels = [x[1] for x in val_examples]
            test_labels = [x[1] for x in test_examples]

            print(f"[CV-{split_i} DATA] for {self.task_name} train size: {len(train_examples)},"
                  f" val size: {len(val_examples)}, test size: {len(test_examples)}, train_ratio: {train_ratio}, "
                  f"is_debug: {self.is_debug}")
            print(f"Train labels: {collections.Counter(train_labels)},"
                  f" val labels: {collections.Counter(val_labels)},"
                  f" test labels :{collections.Counter(test_labels)}")

            if self.task_name == 'churn_predict':
                if self.is_finetune:
                    train_dataset = ChurnPredictDatasetFinetune(train_examples)
                    val_dataset = ChurnPredictDatasetFinetune(val_examples)
                    test_dataset = ChurnPredictDatasetFinetune(test_examples)
                else:
                    train_dataset = ChurnPredictDataset(train_examples,
                                                        embedder=self.embedder,
                                                        save_embedding_cache=self.save_embedding_cache)
                    val_dataset = ChurnPredictDataset(val_examples,
                                                      embedder=self.embedder,
                                                      save_embedding_cache=self.save_embedding_cache)
                    test_dataset = ChurnPredictDataset(all_examples[test_index],
                                                       embedder=self.embedder,
                                                       save_embedding_cache=self.save_embedding_cache)
            else:
                raise NotImplementedError
            print(
                f"Train_val_split index: {split_i},  "
                f"Train size: {len(train_dataset)}, label: {train_dataset.labels}"
                f" Val size: {len(val_dataset)}, label: {val_dataset.labels}"
                f"Test size: {len(test_dataset)}, label: {test_dataset.labels}")
            self._train_val_test_datasets.append((train_dataset, val_dataset, test_dataset))

    def split_train_test_time_series(self, all_examples):
        time_series_split = TimeSeriesSplit(max_train_size=None, n_splits=self.test_n_split)
        print("TimeSeriesSplit")

        train_max_seq_len = train_config.max_seq_len_debug if self.is_debug else train_config.max_seq_len

        for split_i, (train_val_index, test_index) in enumerate(time_series_split.split(all_examples)):

            # Split train into val
            train_val_examples = all_examples[train_val_index]
            train_examples, val_examples = self._split_train_val(train_val_examples)

            # Create tokenizer from Train/val data
            if self.task_name == 'bot_detect':
                if self.use_base_features:
                    tokenizer = WhitespaceTokenizer(self.task_name, is_debug=self.is_debug)
                    tokenizer.create_vocab(
                        [[x_ for x_i, x_ in enumerate(x[0]) if (x_i + 1) % 3 != 0] for x in train_val_examples])
                else:
                    tokenizer = None
            elif self.task_name == 'map_preload':
                tokenizer = WhitespaceTokenizer(self.task_name, is_debug=self.is_debug)
                map_ids = [list(x[1][2]) for x in train_val_examples]
                tokenizer.create_vocab(map_ids, verbose=False)
            else:
                raise Exception

            if self.task_name == 'bot_detect':
                if not self.is_finetune:
                    train_dataset = BotDetectDataset(train_examples,
                                                     embedder=self.embedder,
                                                     tokenizer=tokenizer,
                                                     train_max_seq_len=train_max_seq_len,
                                                     save_embedding_cache=self.save_embedding_cache
                                                     )
                    val_dataset = BotDetectDataset(val_examples,
                                                   embedder=self.embedder,
                                                   tokenizer=tokenizer,
                                                   train_max_seq_len=train_max_seq_len,
                                                   save_embedding_cache=self.save_embedding_cache
                                                   )
                    test_dataset = BotDetectDataset(all_examples[test_index],
                                                    embedder=self.embedder,
                                                    tokenizer=tokenizer,
                                                    train_max_seq_len=train_max_seq_len,
                                                    save_embedding_cache=self.save_embedding_cache
                                                    )
                else:
                    train_dataset = BotDetectDatasetFinetune(train_examples,
                                                             tokenizer=tokenizer,
                                                             train_max_seq_len=train_max_seq_len
                                                             )
                    val_dataset = BotDetectDatasetFinetune(val_examples,
                                                           tokenizer=tokenizer,
                                                           train_max_seq_len=train_max_seq_len
                                                           )
                    test_dataset = BotDetectDatasetFinetune(all_examples[test_index],
                                                            tokenizer=tokenizer,
                                                            train_max_seq_len=train_max_seq_len
                                                            )
            else:
                raise NotImplementedError

            print(
                f"Train_val_split index: {split_i},  "
                f"Train size: {len(train_dataset)}, label: {train_dataset.labels}"
                f" Val size: {len(val_dataset)}, label: {val_dataset.labels}"
                f"Test size: {len(test_dataset)}, label: {test_dataset.labels}")
            self._train_val_test_datasets.append((train_dataset, val_dataset, test_dataset))


class BotDetectTask(GameTask):
    def __init__(self,
                 embedder,
                 task_data_path,
                 test_n_split,
                 debug_N=None,
                 val_ratio=0.1,
                 task_name=None,
                 log_base_dir='',
                 log_save_dir=None,
                 overwrite=False,
                 use_base_features=False,
                 use_pretrain_features=False,
                 is_debug=False,
                 all_examples=None,
                 save_embedding_cache=True,
                 is_finetune=False,
                 do_interpretability=False,
                 player_status_dict=None,
                 raw_seq_max_length=6144
                 ):
        super().__init__(embedder,
                         test_n_split,
                         val_ratio=val_ratio,
                         task_name=task_name,
                         log_base_dir=log_base_dir,
                         log_save_dir=log_save_dir,
                         overwrite=overwrite,
                         debug_N=debug_N,
                         is_debug=is_debug,
                         use_base_features=use_base_features,
                         use_pretrain_features=use_pretrain_features,
                         save_embedding_cache=save_embedding_cache,
                         is_finetune=is_finetune
                         )
        self.raw_seq_max_length = raw_seq_max_length
        if not self.is_result_file_exist:
            if all_examples is None:
                self.all_examples = self._load_all_data(task_data_path)
            else:
                self.all_examples = all_examples
            self.split_train_test_time_series(self.all_examples)
            print(f"Split train/val/test done! Create Datasets done.")

    def _load_all_data(self, data_path):
        # Load hdf5 file
        hdf5_file = h5py.File(data_path, 'r')
        sort_indices = np.array(hdf5_file['sort_indices'])
        print(f"Task-{self.task_name}, read data from {data_path}, size: {len(sort_indices)}")

        if self.debug_N:
            self.debug_N = min(len(sort_indices), self.debug_N)
            sort_indices = sorted(random.sample(list(range(len(sort_indices))), self.debug_N))
            # sort_indices = sort_indices[random_debug_N_indices]

        query_indices = sorted(sort_indices)
        dates = np.array(hdf5_file['date'][query_indices]).astype(str)
        date_objs = [datetime.strptime(x, '%Y-%m-%d') for x in dates]
        sorted_indices = np.argsort(date_objs)

        input = np.array(hdf5_file['input'][query_indices]).astype(str)[sorted_indices]
        assert self.raw_seq_max_length % 3 == 0
        input = input[:, :self.raw_seq_max_length]
        label = np.array(hdf5_file['label'][query_indices]).astype(int)[sorted_indices]

        print(f"[TASK-{self.task_name}] Load hdf5 data done, input_shape: {input.shape}, label_shape: {label.shape}")
        examples = np.array(list(zip(input, label)))

        if not self.is_debug:
            assert len(np.unique(label)) > 1
        return examples


class ChurnPredictTask(GameTask):
    def __init__(self,
                 embedder,
                 task_data_path,
                 test_n_split,
                 debug_N=None,
                 val_ratio=0.1,
                 task_name=None,
                 log_base_dir='',
                 log_save_dir=None,
                 overwrite=False,
                 use_base_features=False,
                 use_pretrain_features=False,
                 is_debug=False,
                 all_examples=None,
                 save_embedding_cache=True,
                 is_finetune=False,
                 player_status_dict=None,
                 do_interpretability=False
                 ):
        super().__init__(embedder,
                         test_n_split,
                         val_ratio=val_ratio,
                         task_name=task_name,
                         log_base_dir=log_base_dir,
                         log_save_dir=log_save_dir,
                         overwrite=overwrite,
                         debug_N=debug_N,
                         is_debug=is_debug,
                         use_base_features=use_base_features,
                         use_pretrain_features=use_pretrain_features,
                         save_embedding_cache=save_embedding_cache,
                         is_finetune=is_finetune
                         )
        self.is_finetune = is_finetune
        self.debug_N = debug_N
        if not self.is_result_file_exist:
            if all_examples is None:
                self.all_examples = self._load_all_data(task_data_path)
            else:
                self.all_examples = all_examples
            self.split_cv(self.all_examples)
            print(f"Split train/val/test done! Create Datasets done.")

    def _load_all_data(self, data_path):

        # Load hdf5 file
        hdf5_file = h5py.File(data_path, 'r')
        all_indices = list(range(hdf5_file['date'].shape[0]))
        print(f"Task-{self.task_name}, read data from {data_path}, size: {len(all_indices)}")

        # <KeysViewHDF5 ['date', 'hand_feature', 'input', 'label']>

        if self.debug_N and self.is_debug:
            all_indices = sorted(random.sample(all_indices, min(len(all_indices), self.debug_N)))

        # TODO, 这里也是按最大的测试数据集来做
        all_indices = sorted(random.sample(all_indices, min(len(all_indices), 1536)))

        # dates = np.array(hdf5_file['date'][all_indices])
        hand_feature = np.array(hdf5_file['hand_feature'][all_indices])
        id_feature = np.array(hdf5_file['input'][all_indices]).astype(str)

        if self.is_finetune:
            if 'longformer' in str(type(self.embedder.model)):
                n_day = 2
            else:
                n_day = 5  # TODO, all set to seven days
            hand_feature = hand_feature[:, -n_day:, :]
            id_feature = id_feature[:, -n_day:, :]

        input = list(zip(id_feature, hand_feature))
        label = np.array(hdf5_file['label'][all_indices]).astype(int)
        label = label[:, 0]
        examples = np.array(list(zip(input, label)))

        label_counter = collections.Counter(label)

        print(f"label_counter: {label_counter}")

        if not self.is_debug:
            assert len(np.unique(label)) > 1
        print(f"[TASK-{self.task_name}] Load hdf5 data done, input_shape: {len(input)}, label_shape: {label.shape}")

        return examples


class ClusteringTask(GameTask):
    def __init__(self,
                 embedder,
                 task_data_path,
                 test_n_split,
                 debug_N=None,
                 val_ratio=0.1,
                 task_name=None,
                 log_base_dir='',
                 log_save_dir=None,
                 overwrite=False,
                 use_base_features=False,
                 use_pretrain_features=False,
                 is_debug=False,
                 all_examples=None,
                 save_embedding_cache=True,
                 is_finetune=False,
                 player_status_dict=None,
                 do_interpretability=False
                 ):
        super().__init__(embedder,
                         test_n_split,
                         val_ratio=val_ratio,
                         task_name=task_name,
                         log_base_dir=log_base_dir,
                         log_save_dir=log_save_dir,
                         overwrite=overwrite,
                         debug_N=debug_N,
                         is_debug=is_debug,
                         use_base_features=use_base_features,
                         use_pretrain_features=use_pretrain_features,
                         save_embedding_cache=save_embedding_cache)
        if not self.is_result_file_exist:
            if all_examples is None:
                self.all_examples = self._load_all_data(task_data_path)
            else:
                self.all_examples = all_examples

    def _load_all_data(self, data_path):
        # Load hdf5 file
        hdf5_file = h5py.File(data_path, 'r')
        all_indices = list(range(hdf5_file['label'].shape[0]))
        print(f"Task-{self.task_name}, read data from {data_path}, size: {len(all_indices)}")

        # <KeysViewHDF5 ['label', 'seq_input']>

        if self.debug_N:
            all_indices = sorted(random.sample(all_indices, min(len(all_indices), self.debug_N)))

        print(f"All keys: {hdf5_file.keys()}")

        seq_input = np.array(hdf5_file['seq_input'][all_indices])
        person_label = np.array(hdf5_file['label'][all_indices]).astype(str)
        # mac_label = np.array(hdf5_file['mac_labels'][all_indices]).astype(str)
        # gender_label = np.array(hdf5_file['gender_labels'][all_indices]).astype(str)
        # grade_label = np.array(hdf5_file['grade_labels'][all_indices]).astype(str)
        # role_class_label = np.array(hdf5_file['role_class_labels'][all_indices]).astype(str)

        print(
            f"[TASK-{self.task_name}] Load hdf5 data done, input_shape: {seq_input.shape}, label_shape: {person_label.shape}")

        return (seq_input, [('person_label', person_label)])


class BuyTimePredictTask(GameTask):
    def __init__(self,
                 embedder,
                 task_data_path,
                 test_n_split,
                 debug_N=None,
                 val_ratio=0.1,
                 task_name=None,
                 log_base_dir='',
                 log_save_dir=None,
                 overwrite=False,
                 use_base_features=False,
                 use_pretrain_features=False,
                 is_debug=False,
                 all_examples=None,
                 save_embedding_cache=True,
                 is_finetune=False,
                 raw_seq_max_length=1026,
                 pos_balance_ratio=0.5,
                 player_status_dict=None,
                 do_interpretability=False
                 ):
        super().__init__(embedder,
                         test_n_split,
                         val_ratio=val_ratio,
                         task_name=task_name,
                         log_base_dir=log_base_dir,
                         log_save_dir=log_save_dir,
                         overwrite=overwrite,
                         debug_N=debug_N,
                         is_debug=is_debug,
                         use_base_features=use_base_features,
                         use_pretrain_features=use_pretrain_features,
                         save_embedding_cache=save_embedding_cache,
                         is_finetune=is_finetune,
                         pos_balance_ratio=pos_balance_ratio,
                         player_status_dict=player_status_dict,
                         do_interpretability=do_interpretability)
        self.pos_balance_ratio = pos_balance_ratio

        if not self.is_result_file_exist:
            if all_examples is None:
                self.all_examples = self._load_all_data(task_data_path, raw_seq_max_length)
            else:
                self.all_examples = all_examples
            self._split_cv_by_role_id(self.all_examples)

    def _load_all_data(self, data_path, raw_seq_max_length):
        hdf5_file = h5py.File(data_path, 'r')
        all_indices = list(range(hdf5_file['label'].shape[0]))
        print(f"Task-{self.task_name}, read data from {data_path}, size: {len(all_indices)}")

        # <KeysViewHDF5 ['item_id', 'item_label', 'label', 'role_id', 'seq_input', 'timegaps']>

        if self.debug_N and self.is_debug:
            all_indices = sorted(random.sample(all_indices, min(len(all_indices), self.debug_N)))

        # TODO，内存塞不下，就用最大测试的DEBUG_N来替代
        all_indices = sorted(random.sample(all_indices, min(len(all_indices), 16384)))

        label = np.array(hdf5_file['label'][all_indices]).astype(int)
        role_id = np.array(hdf5_file['role_id'][all_indices]).astype(str)
        seq_input = np.array(hdf5_file['seq_input'][all_indices]).astype(str)

        dses = []
        for x in seq_input:
            ds = datetime.fromtimestamp(int(x[2])).strftime("%Y-%m-%d")
            dses.append(ds)
        role_id_ds = list(zip(role_id, dses))

        # set the max raw seq length
        seq_input = [x[x != ['[PAD]']][-raw_seq_max_length:] for x in seq_input]

        print(f"[Predict Buy Time] Set max raw sequence length to 1026, label counter: {collections.Counter(label)}")
        timegaps = np.array(hdf5_file['timegaps'][all_indices]).astype(str)
        examples = np.array(list(zip(role_id_ds, seq_input, label)))

        if not self.is_debug:
            assert len(np.unique(label)) > 1
        print(
            f"[TASK-{self.task_name}] Load hdf5 data done,"
            f" input length: {len(seq_input)},"
            f" label_shape: {label.shape},"
            f"label: {collections.Counter(label)}")

        return examples


class UnionRecommendTask(GameTask):
    def __init__(self,
                 embedder,
                 task_data_path,
                 test_n_split,
                 debug_N=None,
                 val_ratio=0.1,
                 task_name=None,
                 log_base_dir='',
                 log_save_dir=None,
                 overwrite=False,
                 use_base_features=False,
                 use_pretrain_features=False,
                 is_debug=False,
                 all_examples=None,
                 save_embedding_cache=True,
                 is_finetune=False,
                 build_data_overwrite=False,
                 player_status_dict=None,
                 do_interpretability=False,
                 ):
        super().__init__(embedder,
                         test_n_split,
                         val_ratio=val_ratio,
                         task_name=task_name,
                         log_base_dir=log_base_dir,
                         log_save_dir=log_save_dir,
                         overwrite=overwrite,
                         debug_N=debug_N,
                         is_debug=is_debug,
                         use_base_features=use_base_features,
                         use_pretrain_features=use_pretrain_features,
                         save_embedding_cache=save_embedding_cache,
                         is_finetune=is_finetune,
                         player_status_dict=player_status_dict
                         )

        # Configs
        # self.use_cpu = False
        self.is_union_embedder_finetune = True

        if not self.is_result_file_exist:
            if all_examples is None:
                self.all_examples = self._load_all_data(task_data_path, build_data_overwrite=build_data_overwrite)
            else:
                self.all_examples = all_examples

            self._split_cv_by_role_id(self.all_examples)
            self._init_embedder()

    def _get_all_union_role_ids(self, union_role_ids, position_dict, raw_seq_hdf5_file,
                                valid_date_range=14, max_role_id=32):
        seqs = []
        for role_date_str, role_id in union_role_ids:

            if len(seqs) >= max_role_id:
                break

            role_date_obj = datetime.strptime(role_date_str, "%Y-%m-%d").date()
            for date_range_i in range(valid_date_range):
                previous_day_obj = role_date_obj - timedelta(days=date_range_i + 1)
                previous_day_str = previous_day_obj.strftime("%Y-%m-%d")
                query_str = f'{role_id}_{previous_day_str}'
                seq_position = position_dict.get(query_str, None)
                if seq_position is not None:
                    seq_position = seq_position.split('#')
                    seq = raw_seq_hdf5_file[seq_position[0]][int(seq_position[1])].astype(str)
                    seqs.append(tuple(seq))
                    break
        seqs = tuple(seqs)
        return seqs

    def _init_embedder(self):
        # force embedder to use CPU
        if hasattr(self.embedder, 'model') and self.is_finetune:
            self.embedder.is_finetune = self.is_finetune
            # if self.use_cpu:
            #     self.embedder.device = torch.device('cpu')
            #     self.embedder.model.to(self.embedder.device)
            #     print(f"change embedder device to CPU")
            if self.is_union_embedder_finetune:
                union_embedder = copy.deepcopy(self.embedder)
                # TODO: 这里不知道什么原因deepcopy会少attribute，embedder.tokenizer下面的vocab之类的都没有copy
                for tokenizer_attribute in self.embedder.tokenizer.__dict__.keys():
                    union_embedder.tokenizer.__dict__[tokenizer_attribute] = self.embedder.tokenizer.__dict__[
                        tokenizer_attribute]

                ignore_layers = ['embeddings', 'layer.0', 'layer.1', 'layer.2', 'layer.3']
                for layer_i, (name, param) in enumerate(union_embedder.model.named_parameters()):
                    is_frozon = False
                    for ignore_layer in ignore_layers:
                        if ignore_layer in name:
                            is_frozon = True
                            break
                    if is_frozon:
                        param.requires_grad = False
                        print(f"Freeze union embedder layer-{name}")
                self.union_embedder = union_embedder
                self.union_embedder.is_finetune = True
            else:
                self.union_embedder = self.embedder
                self.union_embedder.is_finetune = False
        else:
            self.union_embedder = self.embedder
            self.union_embedder.is_finetune = False

    def _load_all_data(self, data_path, build_data_overwrite=False):

        # config
        max_role_id = 16
        max_neg_union_id_N = 5
        max_seq_length = 2049

        save_dir = os.path.dirname(data_path)
        data_pickle_save_name = f'recommend_debug-{self.debug_N}_max_role_id-{max_role_id}'
        data_pickle_save_path = os.path.join(save_dir, data_pickle_save_name)
        data_pickle_paths = glob.glob(
            os.path.join(save_dir, f'{data_pickle_save_name}*'))
        if data_pickle_paths and not build_data_overwrite:
            train_data = []
            for part_data_pickle_path in data_pickle_paths:
                part_train_data = compress_pickle.load(part_data_pickle_path)
                train_data.extend(part_train_data)
                print(f"Load part train data from {part_data_pickle_path}, size: {len(part_train_data)}")
            print(f"Load total train data done, size: {len(train_data)}")
            return train_data

        # data_pickle_save_path = os.path.join(save_dir, f'recommend_debug-{self.debug_N}_max_role_id-{max_role_id}.gz')
        # if os.path.isfile(data_pickle_save_path) and not build_data_overwrite:
        #     train_data = compress_pickle.load(data_pickle_save_path)
        #     print(f"Load train data from {data_pickle_save_path}, size: {len(train_data)}")
        #     return train_data
        else:
            if build_data_overwrite:
                print(f"Overwriting data creation process!!!!!!!!!! Original paths: {data_pickle_paths}")

        # Load hdf5 file
        raw_seq_hdf5_file = h5py.File(data_path, 'r')

        if self.debug_N <= 500:
            raw_seq_hdf5_dict = raw_seq_hdf5_file
        else:
            raw_seq_hdf5_dict = {}
            for hdf5_file_key in raw_seq_hdf5_file.keys():
                data_arr = raw_seq_hdf5_file[hdf5_file_key].value
                raw_seq_hdf5_dict[hdf5_file_key] = data_arr
                print(f"Load all hdf5 data to memory, key: {hdf5_file_key}, data shape: {data_arr.shape}")
            print(f"Load all hdf5 data to memory done!")

        base_dir = os.path.dirname(data_path)

        # position dict
        # 文件丹炉位置: /root/game_bert/data/sample_processed
        position_dict_path = os.path.join(base_dir, 'recommend_role_id_date.json')
        # key: role_id + date
        # key: '565900201_2020-04-05', value: 'nsh_2020-04-05#111',
        position_dict = load_save_json(position_dict_path, 'load')

        # player_union data
        # List, example: ((283, '2020-04-07', 385600283), {420284: 1, 3020283: 0, 4220284: 0})
        # 文件丹炉位置: /root/game_bert/data/recommend_raw/帮会推荐
        player_union_data_path = os.path.join(base_dir, 'player_union_data.pkl')
        player_union_data = pickle.load(open(player_union_data_path, 'rb'))

        # union_members data, key: (235, '2020-04-01', 92820235), value: [('2020-04-01', '600700235'), ...]
        # 文件丹炉位置: /root/game_bert/data/recommend_raw/帮会推荐
        union_members_data_path = os.path.join(base_dir, 'union_members.pkl')
        union_members_data = pickle.load(open(union_members_data_path, 'rb'))

        # Start building data for each player's choice
        train_data = []
        parse_tqdm = tqdm(enumerate(player_union_data), total=len(player_union_data))

        # # force embedder to use CPU
        # if hasattr(embedder, 'model') and self.is_finetune:
        #
        #     if use_cpu:
        #         embedder.device = torch.device('cpu')
        #         embedder.model.to(embedder.device)
        #         print(f"change embedder device to CPU")
        #
        #     if is_union_embedder_finetune:
        #         union_embedder = copy.deepcopy(embedder)
        #         # TODO: 这里不知道什么原因deepcopy会少attribute，embedder.tokenizer下面的vocab之类的都没有copy
        #         for tokenizer_attribute in embedder.tokenizer.__dict__.keys():
        #             union_embedder.tokenizer.__dict__[tokenizer_attribute] = embedder.tokenizer.__dict__[
        #                 tokenizer_attribute]
        #
        #         ignore_layers = ['embeddings', 'layer.0', 'layer.1', 'layer.2', 'layer.3']
        #         for layer_i, (name, param) in enumerate(union_embedder.model.named_parameters()):
        #             is_frozon = False
        #             for ignore_layer in ignore_layers:
        #                 if ignore_layer in name:
        #                     is_frozon = True
        #                     break
        #             if is_frozon:
        #                 param.requires_grad = False
        #                 print(f"Freeze union embedder layer-{name}")
        #         self.union_embedder = union_embedder
        #     else:
        #         union_embedder = embedder
        #         self.union_embedder = None
        # else:
        #     union_embedder = embedder
        #     self.union_embedder = None
        # union_embedder.is_finetune = is_union_embedder_finetune

        for choice_i, choice in parse_tqdm:

            if self.debug_N is not None:
                if len(train_data) > self.debug_N:
                    break

            role_tuple, union_ids_dict = choice

            server_id, role_date_str, role_id = role_tuple

            # Get role sequence data
            role_query_str = f'{role_id}_{role_date_str}'
            role_seq_position = position_dict.get(role_query_str, None)
            if role_seq_position is None:
                # print(f"Skip {choice[0]}, role seq is not Found!")
                continue
            else:
                role_seq_position = role_seq_position.split('#')
                role_seq = raw_seq_hdf5_dict[role_seq_position[0]][int(role_seq_position[1])].astype(str)
                role_seq = role_seq[:max_seq_length]
                role_seq = tuple(role_seq[role_seq != '[PAD]'])

                # if not self.is_finetune:
                #     embedder.is_finetune = False
                # else:
                #     embedder.is_finetune = True

                # role_seq_embedding = embedder.embed([role_seq], batch_size=1, layer=-2, sort=False, verbose=False)[
                #     0].cpu()

            # Load embedding for pos union
            pos_union = [k for k, v in union_ids_dict.items() if v == 1]
            if len(pos_union) != 1:
                continue
            pos_union_id = pos_union[0]
            pos_union_role_ids = tuple(union_members_data[(server_id, role_date_str, pos_union_id)])
            pos_seqs = self._get_all_union_role_ids(pos_union_role_ids, position_dict, raw_seq_hdf5_dict,
                                                    max_role_id=max_role_id)
            if not pos_seqs:
                continue

            # # 这里主要就是
            # pos_seq_embeddings = []
            # for pos_seq in pos_seqs:
            #     pos_seq = pos_seq[:max_seq_length]
            #     pos_seq_embedding = union_embedder.embed([pos_seq], batch_size=1, layer=-2, sort=False,
            #                                              verbose=False).detach().cpu()
            #     pos_seq_embeddings.append(pos_seq_embedding)
            # pos_seq_embeddings = torch.cat(pos_seq_embeddings)
            # pos_seq_embedding = torch.mean(pos_seq_embeddings, dim=0)

            # add positive sample
            train_data.append((role_id,
                               role_date_str,
                               role_seq,
                               (pos_seqs, pos_union_role_ids),
                               1))

            neg_union_ids = [k for k, v in union_ids_dict.items() if v == 0][:max_neg_union_id_N]
            for union_id in neg_union_ids:
                neg_union_role_ids = tuple(union_members_data[(server_id, role_date_str, union_id)])
                neg_seqs = self._get_all_union_role_ids(neg_union_role_ids, position_dict, raw_seq_hdf5_dict,
                                                        max_role_id=max_role_id)
                if not neg_seqs:
                    continue

                # neg_seq_embeddings = []
                # for neg_seq in neg_seqs:
                #     neg_seq = neg_seq[:max_seq_length]
                #     neg_seq_embedding = union_embedder.embed([neg_seq], batch_size=1, layer=-2, sort=False,
                #                                              verbose=False).detach().cpu()
                #     neg_seq_embeddings.append(neg_seq_embedding)
                # neg_seq_embeddings = torch.cat(neg_seq_embeddings)
                # neg_seq_embedding = torch.mean(neg_seq_embeddings, dim=0)

                train_data.append((role_id,
                                   role_date_str,
                                   role_seq,
                                   (neg_seqs, neg_union_role_ids),
                                   0))

            if self.debug_N:
                parse_tqdm.set_description(f"Total complete: {len(train_data)}/{self.debug_N}")
            else:
                parse_tqdm.set_description(f"Total complete: {len(train_data)}")

        # pickle.dump(train_data, open(data_pickle_save_path, 'wb'))
        pickle_size = 32
        pickle_start_index = 0
        part_index = 0
        file_base_dir = os.path.dirname(data_pickle_save_path)
        file_base_name = ntpath.basename(data_pickle_save_path)
        while True:
            temp_dump_data = train_data[pickle_start_index:pickle_start_index + pickle_size]
            temp_dump_file_name = f'{file_base_name}_part{part_index}.gz'
            temp_dump_path = os.path.join(file_base_dir, temp_dump_file_name)
            print(f"Start dumping data (size:{len(temp_dump_data)}) to {temp_dump_path} ...")
            compress_pickle.dump(temp_dump_data, temp_dump_path)
            print(
                f"Save data(size{len(temp_dump_data)})-part{part_index} to {temp_dump_path}")

            pickle_start_index += pickle_size
            part_index += 1
            if pickle_start_index >= len(train_data):
                break

        # compress_pickle.dump(train_data, data_pickle_save_path)
        # print(f"Save data to {data_pickle_save_path}, size: {len(train_data)}")

        train_label_counter = collections.Counter([x[-1] for x in train_data])
        unique_role_ids = set([x[0] for x in train_data])
        print(
            f"Total valid data N: {len(train_data)}, label_counter: {train_label_counter},"
            f" unique_role_id: {len(unique_role_ids)}")
        #
        # # change back to cuda
        # if hasattr(embedder, 'model') and self.is_finetune and use_cpu:
        #     embedder.device = torch.device('cuda')
        #     embedder.model.to(embedder.device)
        #     print(f"change embedder device back to GPU")
        #

        return train_data
