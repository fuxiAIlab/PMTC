import ipdb
import torch
import hashlib
import numpy as np
import collections
from torch.utils.data import Dataset
from datetime import datetime


class DummyEmbedderWithoutPretrain:

    def __init__(self, name='no_pretrain'):
        self.model_name = name
        self.pretrain_task = name
        self.mask_multiple_time = False
        self.max_sequence_length = 0
        self.embedding_dim = 768
        self.use_token_weights = False


def create_collate_fn_churn_predict(embedder=None, batch_size=2):
    def collate_fn_churn_predict(batch):
        batch_hand_feature = []
        batch_transformer_feature = []
        batch_y = []
        for hand_feature, id_seq, y in batch:
            transformer_feature = embedder.embed(id_seq, batch_size=batch_size, layer=-2, sort=False)
            batch_hand_feature.append(hand_feature)
            batch_transformer_feature.append(transformer_feature)
            batch_y.append(y)
        return torch.stack(batch_hand_feature), torch.stack(batch_transformer_feature), torch.stack(batch_y)

    return collate_fn_churn_predict


def create_collate_fn_buy_time_predict(embedder=None, batch_size=2, player_status_dict=None):
    def collate_fn_churn_predict(batch):
        id_seqs = [x[0] for x in batch]
        batch_transformer_features, batch_lengths = embedder.embed(id_seqs,
                                                                   batch_size=batch_size,
                                                                   layer=-2,
                                                                   sort=False,
                                                                   verbose=False,
                                                                   output_mean_pool=False
                                                                   )
        if player_status_dict is not None:
            player_status_features = torch.stack([x[2] for x in batch]).to(embedder.device)
            batch_transformer_features = torch.cat([batch_transformer_features, player_status_features], dim=1)
        batch_y = torch.stack([x[1] for x in batch])
        return (batch_transformer_features, batch_lengths, id_seqs), batch_y

    return collate_fn_churn_predict


def update_union_recommend_embedder_cache(target_seq, target_embedding, embedder):
    if not embedder.is_finetune:
        sample_md5 = compute_md5_for_behavior_seq(target_seq, embedder)
        embedder.update_embedding_cache(sample_md5, target_embedding.detach().cpu())


def get_union_recommend_embedding(target_seq, embedder, batch_size):
    # 这里要注意一下，BPE模型的update_embedding_cache不会做任何事情，他自己会做缓存，是个历史遗留问题
    if not embedder.is_finetune:
        target_seq_md5 = compute_md5_for_behavior_seq(target_seq, embedder)
        if target_seq_md5 in embedder.embedding_cache:
            target_seq_embedding = embedder.embedding_cache[target_seq_md5]
            if hasattr(embedder, 'device'):
                target_seq_embedding = target_seq_embedding.to(embedder.device)
        else:
            target_seq_embedding = embedder.embed([target_seq],
                                                  batch_size=batch_size,
                                                  layer=-2,
                                                  sort=False,
                                                  verbose=False)[0]
            update_union_recommend_embedder_cache(target_seq, target_seq_embedding, embedder)
    else:
        target_seq_embedding = embedder.embed([target_seq],
                                              batch_size=batch_size,
                                              layer=-2,
                                              sort=False,
                                              verbose=False)[0]
    return target_seq_embedding


def create_collate_fn_union_recommend(embedder, union_embedder=None, batch_size=2, player_status_dict=None):
    def collate_fn_churn_predict(batch):
        batch_transformer_features = []
        batch_y = []
        batch_role_id_ds = []
        for (role_seq, (union_seqs, union_role_ids)), y, role_id_ds in batch:

            # get role seq embedding
            role_seq_embedding = get_union_recommend_embedding(role_seq, embedder, batch_size)

            if player_status_dict is not None:
                player_status_feature = torch.from_numpy(player_status_dict[role_id_ds]).float()
                if hasattr(embedder, 'device'):
                    player_status_feature = player_status_feature.to(embedder.device)
                role_seq_embedding = torch.cat([role_seq_embedding, player_status_feature])

            # get union seq embeddings
            union_seq_embeddings = []
            for union_seq, union_role_id in zip(union_seqs, union_role_ids):
                try:
                    union_role_id_ds = (int(union_role_id[1]), union_role_id[0])
                except:
                    print(f"Find invalid union_role_id: {union_role_id}")
                    continue
                union_seq_embedding = get_union_recommend_embedding(union_seq, union_embedder, batch_size)

                if player_status_dict is not None:
                    player_status_feature = torch.from_numpy(player_status_dict[union_role_id_ds]).float()
                    if hasattr(embedder, 'device'):
                        player_status_feature = player_status_feature.to(embedder.device)
                    union_seq_embedding = torch.cat([union_seq_embedding, player_status_feature])
                union_seq_embeddings.append(union_seq_embedding)
            union_seq_embeddings = torch.stack(union_seq_embeddings)
            union_seq_embedding = torch.mean(union_seq_embeddings, dim=0)
            transformer_feature = torch.cat([role_seq_embedding, union_seq_embedding])
            batch_transformer_features.append(transformer_feature)
            batch_y.append(y)
            batch_role_id_ds.append(role_id_ds)
        return torch.stack(batch_transformer_features), torch.stack(batch_y), tuple(batch_role_id_ds)

    return collate_fn_churn_predict


def create_collate_fn_bot_detect(embedder=None, batch_size=1):
    def collate_fn_variable_len(batch):
        batch_x_pad = []
        batch_x_len = []
        batch_transformer_embedding = []
        batch_y = []
        for ((x_pad, x_len), transformer_x, y) in batch:
            batch_x_pad.append(x_pad)
            batch_x_len.append(x_len)

            if embedder is not None:
                # print(f"embedder model train: {embedder.model.training}")
                transformer_x = embedder.embed([transformer_x], batch_size=batch_size, layer=-2, sort=False)
                for x_ in transformer_x:
                    batch_transformer_embedding.append(x_)
            else:
                batch_transformer_embedding.append(transformer_x)
            batch_y.append(y)

        # Sort from longest to shortest
        batch_x_len = torch.stack(batch_x_len)
        if batch_x_len.nelement() == 0:  # Only use transformer features
            batch_x_pad = torch.stack(batch_x_pad)
            batch_y = torch.stack(batch_y)
            batch_transformer_embedding = torch.stack(batch_transformer_embedding)
        else:
            sort_indices = torch.argsort(batch_x_len, descending=True)
            batch_x_pad = torch.stack(batch_x_pad)[sort_indices]
            batch_x_len = batch_x_len[sort_indices]
            batch_transformer_embedding = torch.stack(batch_transformer_embedding)[sort_indices]
            batch_y = torch.stack(batch_y)[sort_indices]

        return ((batch_x_pad, batch_x_len), batch_transformer_embedding, batch_y)

    return collate_fn_variable_len


def convert_timestamp_to_str(timestamp):
    return datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%dT%H:%M:%SZ')


def md5_hash_sample(sample: list,
                    embedder_config: dict):
    assert isinstance(sample, list)
    hash_str = str(sample) + str(sorted(embedder_config.items()))
    return hashlib.md5(hash_str.encode('utf-8')).hexdigest()


def compute_md5_for_behavior_seq(seq, embedder):
    embedder_config = {'use_token_weights': embedder.use_token_weights}
    # embedder_config = {'conca_output_tasks': embedder.conca_output_tasks,
    #                    'mask_multiple_time': embedder.mask_multiple_time,
    #                    'mask_prob': embedder.mask_prob,
    #                    'is_mask_output_concat': embedder.is_mask_output_concat,
    #                    'output_mask_embedding': embedder.output_mask_embedding
    #                    }
    seq_sample_md5 = md5_hash_sample(list(seq), embedder_config)
    return seq_sample_md5


def compute_md5_for_churn_input_seq(seq_example, embedder):
    seq_example_list = [x.tolist() for x in seq_example]
    embedder_config = {'use_token_weights': embedder.use_token_weights}
    # embedder_config = {'conca_output_tasks': embedder.conca_output_tasks,
    #                    'mask_multiple_time': embedder.mask_multiple_time,
    #                    'mask_prob': embedder.mask_prob,
    #                    'is_mask_output_concat': embedder.is_mask_output_concat,
    #                    'output_mask_embedding': embedder.output_mask_embedding
    #                    }
    seq_sample_md5 = md5_hash_sample(seq_example_list, embedder_config)
    return seq_sample_md5


def churn_predict_get_transformer_features(all_examples,
                                           embedder,
                                           is_transformer_finetune,
                                           save_embedding_cache=True):
    """

    Parameters
    ----------
    all_examples
    embedder
    is_transformer_finetune

    Returns:
        Type: torch.Tensor (N x seq_len(14) x dim(768))
    -------

    """
    # Load embedder cache file
    cached_i_features = []
    uncached_i_samples = []

    for example_i, seq_example in enumerate(all_examples):
        seq_sample_md5 = compute_md5_for_churn_input_seq(seq_example, embedder)
        if seq_sample_md5 in embedder.embedding_cache and not is_transformer_finetune:
            cached_i_features.append((example_i, embedder.embedding_cache[seq_sample_md5]))
        else:
            uncached_i_samples.append((example_i, seq_example))

    if uncached_i_samples:

        sample_N = len(uncached_i_samples)
        seq_len = uncached_i_samples[0][1].shape[0]
        uncached_indices = [x[0] for x in uncached_i_samples]
        uncached_samples = [x[1] for x in uncached_i_samples]
        uncached_samples_stack = list(np.concatenate(uncached_samples))
        uncached_samples_stack = [[x_ for x_ in x if x_ != '[PAD]'] for x in uncached_samples_stack]
        print("Start extracting embedding for churn predict ...")

        if 'Longformer' in embedder.model_name or 'Reformer' in embedder.model_name:
            batch_size = 2
        else:
            batch_size = 4

        print(f"batch_size: {batch_size}, max_len: {max([len(x) for x in uncached_samples_stack])}")
        uncached_features = embedder.embed(uncached_samples_stack, batch_size=batch_size, layer=-2, sort=False)
        print("Bert embedding extracted for churn predict done...")
        uncached_features = uncached_features.reshape(sample_N, seq_len, -1).detach().cpu()  # Sample_N x seq_len x 768
        uncached_i_features = list(zip(uncached_indices, uncached_features))
        all_features = cached_i_features + uncached_i_features

        if not is_transformer_finetune:
            # update embedding_cache
            for seq_example, feature in zip(uncached_samples, uncached_features):
                assert feature.shape[0] == 14
                sample_md5 = compute_md5_for_churn_input_seq(seq_example, embedder)
                feature = feature.detach().cpu()
                embedder.update_embedding_cache(sample_md5, feature)

            if save_embedding_cache:
                # Save new torch embedding
                embedder.save_embedding_cache()
    else:
        all_features = cached_i_features

    all_features = sorted(all_features, key=lambda x: x[0])
    # transformer_features = torch.stack([x[1] for x in all_features]) # 这里stack会报cuda memory error
    transformer_features = [x[1] for x in all_features]
    print(f"Load all transformers feature done, len: {len(transformer_features)}")
    return transformer_features


def _get_transformer_features_for_bot_detect_map_preload(time_sorted_examples,
                                                         embedder,
                                                         is_transformer_finetune,
                                                         task_name,
                                                         save_embedding_cache=True,
                                                         ):
    # TODO, 这里的实验设定是这样的，过transformer的特征的长度是模型的长度

    # Load embedder cache file
    cached_i_features = []
    uncached_i_samples = []
    embedder_config = {'use_token_weights': embedder.use_token_weights}

    for example_i, example in enumerate(time_sorted_examples):

        if task_name == 'bot_detect':
            x = example[0]
        elif task_name == 'map_preload':
            _, input, _ = example
            x, _, _ = input
        else:
            raise Exception

        x_no_pad = [x_ for x_ in x if x_ != '[PAD]']
        sample_md5 = md5_hash_sample(x_no_pad, embedder_config)
        if sample_md5 in embedder.embedding_cache and not is_transformer_finetune:
            cached_i_features.append((example_i, embedder.embedding_cache[sample_md5]))
        else:
            uncached_i_samples.append((example_i, x_no_pad))

    if uncached_i_samples:
        uncached_indices = [x[0] for x in uncached_i_samples]
        uncached_samples = [x[1] for x in uncached_i_samples]

        if 'Longformer' in embedder.model_name or 'Reformer' in embedder.model_name:
            batch_size = 2
        else:
            batch_size = 1

        uncached_features = embedder.embed(uncached_samples, batch_size=batch_size, layer=-2).detach().cpu()
        uncached_i_features = list(zip(uncached_indices, uncached_features))
        all_features = cached_i_features + uncached_i_features

        if not is_transformer_finetune:
            # update embedding_cache
            for sample, feature in zip(uncached_samples, uncached_features):
                assert sample[-1] != '[PAD]'
                sample_md5 = md5_hash_sample(sample, embedder_config)
                feature = feature.detach().cpu()
                embedder.update_embedding_cache(sample_md5, feature)

            if save_embedding_cache:
                # Save new torch embedding
                embedder.save_embedding_cache()
    else:
        all_features = cached_i_features

    all_features = sorted(all_features, key=lambda x: x[0])
    transformer_features = torch.stack([x[1] for x in all_features])
    print(f"Load all transformers feature done, shape: {transformer_features.shape}")
    return transformer_features


class BotDetectDataset(Dataset):
    def __init__(self,
                 time_sorted_examples,
                 tokenizer=None,
                 embedder=None,
                 train_max_seq_len=None,
                 is_transformer_finetune=False,
                 save_embedding_cache=True,
                 ):
        # Sort the samples by time
        # [(x_i, y_i), ...]
        self._examples = []
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.labels = []
        self.date_ranges = []
        self.is_transformer_finetune = is_transformer_finetune

        # Load all transformer features
        # TODO, 这里如果要finetune要重新处理下

        if embedder is not None and not isinstance(embedder, DummyEmbedderWithoutPretrain):
            # TODO, 这里的实验设定是这样的，过transformer的特征的长度是模型的长度
            transformer_features = _get_transformer_features_for_bot_detect_map_preload(time_sorted_examples, embedder,
                                                                                        self.is_transformer_finetune,
                                                                                        'bot_detect',
                                                                                        save_embedding_cache=save_embedding_cache)
        else:
            transformer_features = None

        # No Need to compute the max seq length and padding, because padding is already done when saving to hdf5
        for example_i, example in enumerate(time_sorted_examples):
            x, y = example[0], example[1]
            self.labels.append(y)

            if self.tokenizer is not None:
                tokenized_id, no_pad_len, timestamp = self.tokenizer.tokenize_behaviour_seq(x)
                if train_max_seq_len is not None:
                    tokenized_id = tokenized_id[:train_max_seq_len]
                    no_pad_len = min(no_pad_len, train_max_seq_len)
                    timestamp = timestamp[:train_max_seq_len]
                self.date_ranges.extend(timestamp)
                tokenized_id_t = torch.tensor(tokenized_id)
                no_pad_len_t = torch.tensor(no_pad_len)
            else:
                tokenized_id_t = torch.empty(0)
                no_pad_len_t = torch.empty(0)

            if embedder is not None and not isinstance(embedder, DummyEmbedderWithoutPretrain):
                transformer_feature = transformer_features[example_i]
                assert transformer_feature.shape[0] != 0
            else:
                transformer_feature = torch.empty(0)

            self._examples.append(
                ((tokenized_id_t, no_pad_len_t), transformer_feature,
                 torch.tensor([y], dtype=torch.long)))
        self.labels = sorted(list(collections.Counter(self.labels).items()))

        if self.tokenizer is not None:
            self.date_ranges = sorted([int(x) for x in self.date_ranges if x != '[PAD]'])
            self.date_ranges = [convert_timestamp_to_str(self.date_ranges[0]),
                                convert_timestamp_to_str(self.date_ranges[-1])]
        else:
            self.date_ranges = [None, None]

        if 'BpeBow' in str(self.embedder):
            # ------------------------------------------------------------------
            # ALL EXAMPLES
            # ------------------------------------------------------------------
            temp_exampels = torch.stack([x[1] for x in self.examples])
            self.bpe_segment_avg_values = torch.mean(temp_exampels, dim=0)
            # ------------------------------------------------------------------

    @property
    def examples(self):
        return self._examples

    @property
    def size(self):
        return self.__len__()

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class BotDetectDatasetFinetune(Dataset):
    def __init__(self,
                 time_sorted_examples,
                 tokenizer=None,
                 train_max_seq_len=None
                 ):
        # Sort the samples by time
        # [(x_i, y_i), ...]
        self._examples = []
        self.tokenizer = tokenizer
        self.labels = []
        self.date_ranges = []

        # No Need to compute the max seq length and padding, because padding is already done when saving to hdf5
        for example_i, example in enumerate(time_sorted_examples):
            x, y = example[0], example[1]
            no_pad_x = tuple([log_id for log_id in x if log_id != '[PAD]'])
            self.labels.append(y)

            if self.tokenizer is not None:
                tokenized_id, no_pad_len, timestamp = self.tokenizer.tokenize_behaviour_seq(x)
                if train_max_seq_len is not None:
                    tokenized_id = tokenized_id[:train_max_seq_len]
                    no_pad_len = min(no_pad_len, train_max_seq_len)
                    timestamp = timestamp[:train_max_seq_len]
                self.date_ranges.extend(timestamp)
                tokenized_id_t = torch.tensor(tokenized_id)
                no_pad_len_t = torch.tensor(no_pad_len)
            else:
                tokenized_id_t = torch.empty(0)
                no_pad_len_t = torch.empty(0)
            self._examples.append(
                ((tokenized_id_t, no_pad_len_t), no_pad_x,
                 torch.tensor([y], dtype=torch.long)))
        self.labels = sorted(list(collections.Counter(self.labels).items()))

        if self.tokenizer is not None:
            self.date_ranges = sorted([int(x) for x in self.date_ranges if x != '[PAD]'])
            self.date_ranges = [convert_timestamp_to_str(self.date_ranges[0]),
                                convert_timestamp_to_str(self.date_ranges[-1])]
        else:
            self.date_ranges = [None, None]

    @property
    def examples(self):
        return self._examples

    @property
    def size(self):
        return self.__len__()

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class ChurnPredictDatasetFinetune(Dataset):
    def __init__(self, all_examples):
        # Sort the samples by time
        # [(x_i, y_i), ...]
        self._examples = []
        self.labels = []
        self.date_ranges = [None, None]

        # No Need to compute the max seq length and padding, because padding is already done when saving to hdf5
        for example_i, (input_x, y) in enumerate(all_examples):
            self.labels.append(y)
            id_seq, hand_feature = input_x

            # id_seq: 14 x 6144
            # transformer_feature: 14 x 768
            # hand_feature:14 x 194
            # y: [1]
            id_seq = tuple([tuple([x_ for x_ in x if x_ != '[PAD]']) for x in id_seq])
            self._examples.append((torch.tensor(hand_feature).float(), id_seq, torch.tensor([y])))
        self.labels = sorted(list(collections.Counter(self.labels).items()))

    @property
    def examples(self):
        return self._examples

    @property
    def size(self):
        return self.__len__()

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class ChurnPredictDataset(Dataset):
    def __init__(self,
                 all_examples,
                 embedder=None,
                 is_transformer_finetune=False,
                 save_embedding_cache=True
                 ):
        # Sort the samples by time
        # [(x_i, y_i), ...]
        self._examples = []
        self.embedder = embedder
        self.labels = []
        self.is_transformer_finetune = is_transformer_finetune
        self.date_ranges = [None, None]

        # Load all transformer features
        # TODO, 这里如果要finetune要重新处理下

        if embedder is not None and not isinstance(embedder, DummyEmbedderWithoutPretrain):
            id_seqs = []
            for input_x, _ in all_examples:
                id_seq, _ = input_x
                id_seqs.append(id_seq)
            transformer_features = churn_predict_get_transformer_features(id_seqs,
                                                                          embedder,
                                                                          self.is_transformer_finetune,
                                                                          save_embedding_cache=save_embedding_cache)
        else:
            transformer_features = None

        # No Need to compute the max seq length and padding, because padding is already done when saving to hdf5
        for example_i, (input_x, y) in enumerate(all_examples):

            self.labels.append(y)
            id_seq, hand_feature = input_x

            if embedder is not None and not isinstance(embedder, DummyEmbedderWithoutPretrain):
                transformer_feature = transformer_features[example_i]
                assert transformer_feature.shape[0] != 0
            else:
                transformer_feature = torch.empty(0)

            # transformer_feature: 14 x 768
            # hand_feature:14 x 194
            # y: [1]

            self._examples.append((torch.tensor(hand_feature).float(), transformer_feature, torch.tensor([y])))
        self.labels = sorted(list(collections.Counter(self.labels).items()))

        if 'BpeBow' in str(self.embedder):
            # ------------------------------------------------------------------
            # ALL EXAMPLES
            # ------------------------------------------------------------------
            temp_exampels = torch.cat([x[1] for x in self.examples])
            self.bpe_segment_avg_values = torch.mean(temp_exampels, dim=0)
            # ------------------------------------------------------------------

    @property
    def examples(self):
        return self._examples

    @property
    def size(self):
        return self.__len__()

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class BuyTimePredictDataset(Dataset):
    def __init__(self,
                 examples,
                 embedder=None,
                 player_status_dict=None
                 ):
        # examples[0]: role_id, seq_array, label

        seqs = [list(x[1]) for x in examples]
        seq_embeddings = embedder.embed(seqs, batch_size=4, layer=-2, sort=False, to_cpu=True)

        if player_status_dict is not None:
            player_status_features = []
            for x in examples:
                player_status_features.append(player_status_dict[(int(x[0][0]), x[0][1])])
            player_status_features = torch.from_numpy(np.stack(player_status_features)).float()

            if hasattr(embedder, 'device'):
                player_status_features = player_status_features.to(embedder.device)
            new_seq_embeddings = torch.cat([seq_embeddings, player_status_features], dim=1)
        else:
            new_seq_embeddings = seq_embeddings

        labels = [x[2] for x in examples]
        labels = torch.tensor(labels).view(-1, 1)

        self._examples = []
        for seq_embedding, label in zip(new_seq_embeddings, labels):
            # (batch_transformer_features, batch_lengths, id_seqs)
            self._examples.append(((seq_embedding, torch.empty(0), torch.empty(0)), label))

        # get label counter
        self.labels = sorted(list(collections.Counter(labels.flatten().tolist()).items()))
        self.date_ranges = None

    @property
    def examples(self):
        return self._examples

    @property
    def size(self):
        return self.__len__()

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class BuyTimePredictFinetuneDataset(Dataset):
    def __init__(self, examples, player_status_dict=None):
        # examples[0]: role_id, seq_array, label
        self._examples = []

        seqs = [tuple(x[1]) for x in examples]
        labels = [x[2] for x in examples]
        labels = torch.tensor(labels).view(-1, 1)

        if player_status_dict is not None:
            player_status_features = []
            for x in examples:
                player_status_features.append(player_status_dict[(int(x[0][0]), x[0][1])])
            player_status_features = torch.from_numpy(np.stack(player_status_features)).float()
        else:
            player_status_features = torch.empty_like(labels)

        self._examples = []
        for seq, label, player_status_feature in zip(seqs, labels, player_status_features):
            self._examples.append((seq, label, player_status_feature))

        # get label counter
        self.labels = sorted(list(collections.Counter(labels.flatten().tolist()).items()))
        self.date_ranges = None

    @property
    def examples(self):
        return self._examples

    @property
    def size(self):
        return self.__len__()

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class UnionRecommendDataset(Dataset):
    def __init__(self,
                 examples
                 ):
        self._examples = []
        labels = []

        for role_id, date_str, role_seq, union_seqs, label in examples:
            self._examples.append(((role_seq, union_seqs), torch.tensor([label]), (role_id, date_str)))
            labels.append(label)

        # get label counter
        self.labels = sorted(list(collections.Counter(labels).items()))
        self.date_ranges = None

    @property
    def examples(self):
        return self._examples

    @property
    def size(self):
        return self.__len__()

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, idx):
        return self.examples[idx]
