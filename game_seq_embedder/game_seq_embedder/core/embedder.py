import os
import copy
import time
import ipdb
import dill
import torch
import collections
import functools
import hashlib
from datetime import datetime
import numpy as np

from tqdm import tqdm
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from ..transformers.modeling_utils import PreTrainedModel
from .utils import mask_prob_masking_post_process
from .bert_tokenizer_custom import WhiteSpaceTokenizer


def cut_by_batchsize(iter_object, batch_size):
    cut_indices = [i for i, _ in enumerate(iter_object)][::batch_size]
    if cut_indices[-1] != len(iter_object):
        cut_indices.append(len(iter_object))
    cut_indices = list(zip(cut_indices[:-1], cut_indices[1:]))
    return cut_indices


def padding_batchX(X, pad_index):
    # do padding
    pad_len = max([len(x) for x in X])
    pad_X = []
    attention_mask_X = []
    for seq_x in X:
        unpad_len = pad_len - len(seq_x)
        attention_mask = [1] * len(seq_x)
        seq_x.extend([pad_index] * unpad_len)
        attention_mask.extend([0] * unpad_len)
        attention_mask_X.append(attention_mask)
        pad_X.append(seq_x)

    attention_mask_X = torch.tensor(attention_mask_X, dtype=torch.long)

    pad_X = torch.tensor(pad_X, dtype=torch.long)

    return pad_X, attention_mask_X


def _embedding_postprocess(embeddings, to_numpy, to_float64, to_cpu):
    if to_numpy and isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
        if to_float64:
            embeddings = embeddings.astype(np.float64)
    else:
        if to_cpu:
            embeddings = embeddings.detach().cpu()
    return embeddings


def compute_module_params_hash(module):
    params = tuple(str(p) for p in module.parameters())
    params_str = functools.reduce(lambda a, b: a + b, params)
    md5_str = hashlib.md5(params_str.encode('utf-8')).hexdigest()
    return md5_str


class BehaviorSequenceEmbedder:

    def __init__(self,
                 tokenizer: Union[WhiteSpaceTokenizer],
                 model: PreTrainedModel,
                 behave_tokenizer: Union[WhiteSpaceTokenizer] = None,
                 design_tokenizer: Union[WhiteSpaceTokenizer] = None,
                 use_time_embed: bool = False,
                 use_sinusoidal: bool = False,
                 is_finetune: bool = False,
                 seperate_design_id: bool = False,
                 use_bpe: bool = False,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 embedding_cache_dir: str = None,
                 token_weights: torch.Tensor = None
                 ):
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.model_name = model._get_name()

        # # Decide whether it is multitask model or not
        # self.is_model_multitask = True if 'BertForMaskedLMTimeEmbedMultiTask' in str(self.model) else False

        if is_finetune:
            print(f"Set {self.model_name} to fine-tune mode")
            self.model.train()
        else:
            print(f"Set {self.model_name} to embedding extraction mode")
            self.model.eval()

        # Compute model hash
        self.model_hash = compute_module_params_hash(self.model)

        self.is_finetune = is_finetune

        self.use_time_embed = use_time_embed
        self.use_sinusoidal = use_sinusoidal
        self.seperate_design_id = seperate_design_id
        self.use_bpe = use_bpe
        self.device = device
        self.behave_tokenizer = behave_tokenizer
        self.design_tokenizer = design_tokenizer
        self.output_mask_embedding = True
        self.token_weights = token_weights

        if self.is_finetune:
            self.use_token_weights = False
        else:
            if self.token_weights is not None:
                self.use_token_weights = True
            else:
                self.use_token_weights = False

        # embedding cache
        if embedding_cache_dir:
            self.embedding_cache_path = os.path.join(embedding_cache_dir, f'{self.model_hash}.pkl')
        else:
            self.embedding_cache_path = ''
        self._embedding_cache = None
        self.is_embedding_loaded = False

        # Set embedding dim
        # self.conca_output_tasks = conca_output_tasks

        if self.model_name != 'ReformerModelWithLMHead':
            self.embedding_dim = 768
        else:
            self.embedding_dim = 192

        # Get Max sequence Length
        self.max_sequence_length = None

        if self.model_name == 'ReformerModelWithLMHead':
            for params_name, params in self.model.named_parameters():
                if params_name == 'reformer.embeddings.position_embeddings.weights.0':
                    self.max_sequence_length = params.shape[0] * params.shape[2]
                    break
        else:
            for param_name, param in self.model.named_parameters():
                if 'position_embeddings' in param_name:
                    self.max_sequence_length = param.shape[0]
                    break

        assert self.max_sequence_length

        # read model name
        model_name = self.model._get_name()

        # TODO work around for longformer postion embedding bug
        if model_name == 'LongformerForMaskedLM':
            self.max_sequence_length -= 2

        # set tuple index to read from hugging face output
        if model_name == 'ReformerModelWithLMHead':
            self.read_index = 2
        else:
            self.read_index = 1

        print(f"Max sequence length: {self.max_sequence_length}")

        self.max_input_id_index = None
        for param_name, param in self.model.named_parameters():
            if 'word_embeddings' in param_name:
                self.max_input_id_index = param.shape[0] - 1
                break
        print(f"Max word embedding index: {self.max_input_id_index}")

        # Get Max time index
        if self.use_time_embed:
            if not use_sinusoidal:
                # compute max time index
                for param_name, param in self.model.named_parameters():
                    if 'time_gap_embeddings' in param_name:
                        self.max_time_index = param.shape[0] - 1
                        break
                print(f"Max time index: {self.max_time_index}")
            else:
                self.max_time_index = None
        else:
            self.max_time_index = None

    def load_embedding_cache(self):
        if os.path.isfile(self.embedding_cache_path):
            print(
                f"[Embedding Cache] Try Load embedding cache from {self.embedding_cache_path} ...")
            with open(self.embedding_cache_path, 'rb') as read_file:
                embedding_cache = dill.load(read_file)
            print(
                f"[Embedding Cache] Load embedding cache from {self.embedding_cache_path} done, size: {len(embedding_cache)}")
        else:
            embedding_cache = {}
            print(f"P[Embedding Cache] path-{self.embedding_cache_path} not found, init new embedding cache!")
        self.is_embedding_loaded = True
        self._embedding_cache = embedding_cache

    def save_embedding_cache(self):
        if self.is_embedding_loaded:
            assert self._embedding_cache
            with open(self.embedding_cache_path, 'wb') as dump_file:
                dill.dump(self._embedding_cache, dump_file)
            print(f"[Embedding Cache] Save embedding cache to {self.embedding_cache_path}, "
                  f"size: {len(self._embedding_cache)}")
        else:
            raise Exception("[Embedding Cache] Embedding cache is not Loaded!!! Please Load before saving!")

    def update_embedding_cache(self,
                               key: str,
                               value: torch.Tensor):
        if key in self._embedding_cache:
            assert bool(torch.mean(value - self._embedding_cache[key]) < 1e-7), ipdb.set_trace()
        else:
            self._embedding_cache[key] = value

    @property
    def embedding_cache(self):
        return self._embedding_cache

    @property
    def model_params(self):
        return self.model.parameters()

    def set_to_feature_extration_mode(self):
        self.is_finetune = False
        self.model.eval()

    def _cut_by_max_seq_length(self, input_ids, attention_masks, time_gaps=None, design_ids=None):
        input_ids = input_ids[:, :self.max_sequence_length]
        attention_masks = attention_masks[:, :self.max_sequence_length]

        if design_ids is not None:
            design_ids = design_ids[:, :self.max_sequence_length]
            assert input_ids.shape == attention_masks.shape == design_ids.shape

        if time_gaps is not None:
            time_gaps = time_gaps[:, :self.max_sequence_length]
            assert input_ids.shape == attention_masks.shape == time_gaps.shape
        else:
            assert input_ids.shape == attention_masks.shape

        return input_ids, design_ids, attention_masks, time_gaps

    def _compute_time_gaps(self, seq_time_stamps, use_sinusoidal):
        """

        :param seq_time_stamps:
        # ['1604578001', '1604578002', '1604578003', '1604578004', '1604123123']
        :return:
        """
        # TODO, check timestamp is valid
        # TODO, check time stamp previous smaller than later
        seq_time_stamps = torch.tensor([int(x) for x in seq_time_stamps])

        if use_sinusoidal:
            time_gap0 = datetime.fromtimestamp(int(seq_time_stamps[0]))
            today_start = datetime(year=time_gap0.year, month=time_gap0.month, day=time_gap0.day)
            today_start_timestamp = int(time.mktime(today_start.timetuple()))
            seq_time_stamps = seq_time_stamps - today_start_timestamp
            return seq_time_stamps
        else:
            # compute time gaps
            seq_time_gaps = torch.full_like(seq_time_stamps, fill_value=0)
            seq_time_gaps[:len(seq_time_stamps[1:])] = seq_time_stamps[1:] - seq_time_stamps[:-1]
            seq_time_gaps = seq_time_gaps / 100  # base unit 0.1s
            return seq_time_gaps

    def _pad_game_ids_bpe(self, cut_sequences):

        assert self.use_bpe
        max_seq_length = max([len(x) for x in cut_sequences])
        bpe_max_seq_len = 0

        design_ids = None
        input_ids = torch.full((len(cut_sequences), max_seq_length), self.tokenizer.vocab['[PAD]'],
                               dtype=torch.long)
        attention_masks = torch.full((len(cut_sequences), max_seq_length), 0, dtype=torch.long)

        time_gaps = None

        for seq_i, game_ids in enumerate(cut_sequences):

            try:
                unk_cn_char = self.tokenizer.game_id_cn_char_map['<unk>']
            except:
                raise Exception("<unk> is not a valid key for game id cn char")

            game_ids_cn_chars = ''.join(
                [self.tokenizer.game_id_cn_char_map.get(x, unk_cn_char) for x in
                 game_ids if x != '[PAD]'])
            tokenizer_output = self.tokenizer.encode(game_ids_cn_chars)
            game_indices = tokenizer_output.ids
            game_tokens = tokenizer_output.tokens
            bpe_max_seq_len = max(bpe_max_seq_len, len(game_indices))
            input_ids[seq_i][:len(game_indices)] = torch.tensor(game_indices)
            attention_masks[seq_i][:len(game_indices)] = 1

        input_ids, attention_masks = input_ids[:, :bpe_max_seq_len], attention_masks[:, :bpe_max_seq_len]

        input_ids, design_ids, attention_masks, time_gaps = self._cut_by_max_seq_length(input_ids,
                                                                                        attention_masks,
                                                                                        time_gaps=time_gaps,
                                                                                        design_ids=design_ids)

        return input_ids, design_ids, attention_masks, time_gaps

    def _pad_game_id_and_time_gap(self,
                                  cut_sequences,
                                  use_time_embed,
                                  use_bpe,
                                  seperate_design_id,
                                  use_sinusoidal):

        if seperate_design_id:
            assert not use_bpe

        max_seq_length = max([int(len(x) * 2 / 3) for x in cut_sequences])
        bpe_max_seq_len = 0

        if seperate_design_id:
            tensor_len = int(max_seq_length / 2)
            input_ids = torch.full((len(cut_sequences), tensor_len),
                                   self.behave_tokenizer.vocab['[PAD]'],
                                   dtype=torch.long)
            design_ids = torch.full((len(cut_sequences), tensor_len),
                                    self.design_tokenizer.vocab['[PAD]'],
                                    dtype=torch.long)
            attention_masks = torch.full((len(cut_sequences), tensor_len), 0, dtype=torch.long)
            time_gaps = torch.full((len(cut_sequences), tensor_len), -1, dtype=torch.long)
        else:
            design_ids = None
            input_ids = torch.full((len(cut_sequences), max_seq_length), self.tokenizer.vocab['[PAD]'],
                                   dtype=torch.long)
            attention_masks = torch.full((len(cut_sequences), max_seq_length), 0, dtype=torch.long)

            if use_time_embed:
                time_gaps = torch.full((len(cut_sequences), max_seq_length), -1, dtype=torch.long)
            else:
                time_gaps = None

        for seq_i, seq in enumerate(cut_sequences):

            assert len(seq) % 3 == 0

            game_ids = []
            for id_i, game_id in enumerate(seq):
                if (id_i + 1) % 3 != 0:
                    game_ids.append(game_id)

            if use_bpe:
                game_ids_cn_chars = ''.join(
                    [self.tokenizer.game_id_cn_char_map.get(x, self.tokenizer.game_id_cn_char_map['[UNK]']) for x in
                     game_ids])
                tokenizer_output = self.tokenizer.encode(game_ids_cn_chars)
                game_indices = tokenizer_output.ids
                game_tokens = tokenizer_output.tokens

                # # TODO assert , remove in future
                # assert len(game_ids_cn_chars) == len(''.join(game_tokens).replace('▁', '')), ipdb.set_trace()

                bpe_max_seq_len = max(bpe_max_seq_len, len(game_indices))

                # TODO, reformer padding problem when training
                if use_time_embed:
                    seq_time_stamps = seq[2::3]
                    seq_time_gaps = self._compute_time_gaps(seq_time_stamps, False)
                    seq_time_gaps_double = time_gaps[seq_i].clone().detach()
                    seq_time_gaps_double[::2][:len(seq_time_gaps)] = seq_time_gaps.clone().detach()
                    seq_time_gaps_double[1::2][:len(seq_time_gaps)] = seq_time_gaps.clone().detach()

                    start_index = 0
                    time_gap_one_seq = []
                    for game_i, game_subword in enumerate(game_tokens):
                        game_subword = game_subword.replace('_', '').replace('▁', '')
                        end_index = start_index + len(game_subword)
                        time_gap = int(torch.sum(seq_time_gaps_double[start_index:end_index]))
                        time_gap_one_seq.append(time_gap)
                        start_index += len(game_subword)
                    time_gaps[seq_i][:len(time_gap_one_seq)] = torch.tensor(time_gap_one_seq)

                input_ids[seq_i][:len(game_indices)] = torch.tensor(game_indices)
                attention_masks[seq_i][:len(game_indices)] = 1

            else:
                if seperate_design_id:
                    behave_tokenized_ids = self.behave_tokenizer.encode(' '.join(game_ids[::2]))['ids']
                    design_tokenized_ids = self.design_tokenizer.encode(' '.join(game_ids[1::2]))['ids']
                    input_ids[seq_i][:len(behave_tokenized_ids)] = torch.tensor(behave_tokenized_ids)
                    design_ids[seq_i][:len(design_tokenized_ids)] = torch.tensor(design_tokenized_ids)
                    attention_masks[seq_i][:len(behave_tokenized_ids)] = 1
                else:
                    game_indices = self.tokenizer.encode(' '.join(game_ids))['ids']
                    input_ids[seq_i][:len(game_indices)] = torch.tensor(game_indices)
                    attention_masks[seq_i][:len(game_indices)] = 1

                if use_time_embed:
                    seq_time_stamps = seq[2::3]

                    # compute time gaps
                    seq_time_gaps = self._compute_time_gaps(seq_time_stamps, use_sinusoidal)
                    assert seq_time_gaps.shape[0] == len(seq_time_stamps)
                    if seperate_design_id:
                        assert len(behave_tokenized_ids) == len(seq_time_gaps)
                    else:
                        assert len(game_indices) == 2 * len(seq_time_gaps)

                    if seperate_design_id:
                        time_gaps[seq_i][:len(seq_time_gaps)] = seq_time_gaps
                    else:
                        time_gaps[seq_i][::2][:len(seq_time_gaps)] = seq_time_gaps
                        time_gaps[seq_i][1::2][:len(seq_time_gaps)] = seq_time_gaps

        if use_bpe:
            input_ids, attention_masks = input_ids[:, :bpe_max_seq_len], attention_masks[:, :bpe_max_seq_len]
            if use_time_embed:
                time_gap_max_seq_len = max([len(x[x >= 0]) for x in time_gaps])
                assert time_gap_max_seq_len == bpe_max_seq_len  # TODO, del in the future
                time_gaps = time_gaps[:, :bpe_max_seq_len]

        if use_time_embed:
            if use_sinusoidal:
                pass
            else:
                time_gaps[time_gaps > self.max_time_index] = self.max_time_index
                time_gaps[time_gaps == -1] = 0

        input_ids, design_ids, attention_masks, time_gaps = self._cut_by_max_seq_length(input_ids,
                                                                                        attention_masks,
                                                                                        time_gaps=time_gaps,
                                                                                        design_ids=design_ids)

        return input_ids, design_ids, attention_masks, time_gaps

    def mean_pool_with_mask(self, batch_layer_hiddens, attention_masks):
        embedding = []
        for batch_i, layer_hidden in enumerate(batch_layer_hiddens):
            # layer_hidden, shape: seq_len x 768
            layer_hidden = layer_hidden[attention_masks[batch_i].to(bool)]
            embedding.append(torch.mean(layer_hidden, dim=0))
        embedding = torch.stack(embedding)
        return embedding

    def filter_with_attention_mask(self, batch_layer_hiddens, attention_masks):
        embedding = []
        for batch_i, layer_hidden in enumerate(batch_layer_hiddens):
            # layer_hidden, shape: seq_len x 768
            layer_hidden = layer_hidden[attention_masks[batch_i].to(bool)]
            embedding.append(layer_hidden)
        return embedding

    def padding_embedding(self, batch_layer_hiddens, attention_masks):
        embedding = []
        for batch_i, layer_hidden in enumerate(batch_layer_hiddens):
            # layer_hidden, shape: seq_len x 768
            embedding.append(layer_hidden)
        embedding = torch.stack(embedding)
        return embedding

    def embed_inner_helper(self, input_ids, design_ids, attention_masks, time_gaps, layer,
                           output_mean_pool=True):
        # some assertions
        assert input_ids.shape == attention_masks.shape
        assert max(input_ids.flatten()) <= self.max_input_id_index

        if time_gaps is not None:
            assert input_ids.shape == time_gaps.shape
            time_gaps = time_gaps.to(self.device)
            if design_ids is not None:
                assert input_ids.shape == design_ids.shape
                design_ids = design_ids.to(self.device)
                batch_layer_hiddens = self.model(input_ids=input_ids,
                                                 design_ids=design_ids,
                                                 attention_mask=attention_masks,
                                                 time_gaps=time_gaps,
                                                 output_hidden_states=True,
                                                 )[self.read_index][layer]
            else:
                batch_layer_hiddens = self.model(input_ids=input_ids,
                                                 attention_mask=attention_masks,
                                                 time_gaps=time_gaps,
                                                 output_hidden_states=True,
                                                 )[self.read_index][layer]
        else:
            batch_layer_hiddens = self.model(input_ids=input_ids,
                                             attention_mask=attention_masks,
                                             output_hidden_states=True)[self.read_index][layer]

        # if self.token_weights is not None and not self.is_finetune:
        #     # expand_token_vocab_weights : batch_size x vocab_size
        #     expand_token_vocab_weights = self.token_weights.expand(batch_layer_hiddens.shape[0],
        #                                                            self.token_weights.shape[0]).to(self.device)
        #     # batch_gathered_weights: batch_size x batch_max_seq_len, the same shape as input_ids
        #     batch_gathered_weights = expand_token_vocab_weights.gather(1, input_ids)
        #     batch_gathered_weights = batch_gathered_weights.unsqueeze(-1)
        #     # expand to the same shape as batch_layer_hiddens
        #     batch_gathered_weights = batch_gathered_weights.expand(-1, -1, batch_layer_hiddens.shape[-1])
        #     batch_layer_hiddens = batch_layer_hiddens * batch_gathered_weights

        assert input_ids.shape[0] == batch_layer_hiddens.shape[0]

        # Todo, figure out why longformer's output length is longer than the input, e.g., if the input length is
        # 1904, the output would be 1920

        if self.model_name == 'LongformerForMaskedLM':
            batch_layer_hiddens = batch_layer_hiddens[:, :attention_masks.shape[1], :]
        elif self.model_name == 'ReformerModelWithLMHead':
            batch_layer_hiddens = batch_layer_hiddens[:, :attention_masks.shape[1], :]
        else:
            assert batch_layer_hiddens.shape[1] == attention_masks.shape[1], ipdb.set_trace()

        if output_mean_pool:
            embedding = self.mean_pool_with_mask(batch_layer_hiddens, attention_masks)
        else:
            no_pad_lengths = [len(x[x != self.tokenizer.vocab['[PAD]']]) for x in input_ids]
            embedding = [x[:no_pad_lengths[x_i]] for x_i, x in enumerate(batch_layer_hiddens)]
            # embedding = self.padding_embedding(batch_layer_hiddens, attention_masks)
        return embedding

    def _mask_output_embedding(self, embedding, masked_indices, attention_masks):
        """
        Only output embedding of Non [MASK] tokens
        Returns

        embedding: batch_size x max_seq_len x 768
        masked_indices: batch_size x max_seq_len
        -------
        """
        masked_embeddings = []
        for i, (embedding_, embedding_mask) in enumerate(zip(embedding, masked_indices)):
            embedding_mask = ~embedding_mask & attention_masks[i].bool().cpu()
            masked_embedding = torch.mean(embedding_[embedding_mask], dim=0)
            if torch.isnan(masked_embedding).any():
                ipdb.set_trace()
            masked_embeddings.append(masked_embedding)
        masked_embeddings = torch.stack(masked_embeddings, dim=0)
        return masked_embeddings

    def embed_inner(self,
                    sequences: List[List[str]],
                    batch_size: int = 4,
                    layer: int = -2,
                    verbose: bool = True,
                    output_mean_pool: bool = True,
                    to_cpu: bool = False,
                    is_general_sequence: bool = False  # 这里指的就是是不是普通的序列，类似于任意游戏的序列，不包含时间戳，也没有规律
                    ):

        # get texts hash
        cut_indices = cut_by_batchsize(sequences, batch_size)
        embeddings = []
        cut_indices_tqdm = tqdm(cut_indices, total=len(cut_indices), disable=True if not verbose else False)
        for cut_index in cut_indices_tqdm:
            cut_sequences = sequences[cut_index[0]: cut_index[1]]
            if is_general_sequence:
                input_ids, design_ids, attention_masks, time_gaps = self._pad_game_ids_bpe(cut_sequences)
            else:
                input_ids, design_ids, attention_masks, time_gaps = self._pad_game_id_and_time_gap(cut_sequences,
                                                                                                   self.use_time_embed,
                                                                                                   self.use_bpe,
                                                                                                   self.seperate_design_id,
                                                                                                   self.use_sinusoidal
                                                                                                   )
            attention_masks = attention_masks.to(self.device)
            input_ids = input_ids.to(self.device)
            embedding = self.embed_inner_helper(input_ids, design_ids, attention_masks, time_gaps, layer,
                                                output_mean_pool=output_mean_pool)
            if output_mean_pool:
                embeddings.append(embedding)
            else:
                if to_cpu:
                    embedding = [x.to('cpu') for x in embedding]
                embeddings.extend(embedding)

        if output_mean_pool:
            embeddings = torch.cat(embeddings, dim=0)
        return embeddings

    def _reformer_padding_input(self, sequences):
        for name, params in self.model.named_parameters():
            if name == 'reformer.embeddings.position_embeddings.weights.0':
                reformer_max_seq_len = params.shape[0] * params.shape[2]
                break
        reformer_pad_len = reformer_max_seq_len * 3 / 2
        sequences_reformer_pad = []
        for seq in sequences:
            to_pad_seq = [self.tokenizer._pad_token for _ in range(int(reformer_pad_len - len(seq)))]
            if to_pad_seq:
                to_pad_seq = np.array(to_pad_seq)
                to_pad_seq[2::3] = -1
                to_pad_seq = list(copy.deepcopy(seq)) + list(to_pad_seq)
            else:
                to_pad_seq = copy.deepcopy(seq)
            sequences_reformer_pad.append(to_pad_seq)
        sequences = sequences_reformer_pad
        return sequences

    def _pad_embeddings(self, embeddings, to_cpu):
        device = 'cpu' if to_cpu else self.device
        embedding_lengths = [x.shape[0] for x in embeddings]
        max_len = max(embedding_lengths)
        pad_embeddings = torch.zeros((len(embedding_lengths), max_len, embeddings[0].shape[1])).to(device)
        for i, pad_embedding in enumerate(pad_embeddings):
            pad_embedding[:embedding_lengths[i], :] = embeddings[i]
        return pad_embeddings, embedding_lengths

    def embed(self,
              sequences: List[List[str]],
              batch_size: int = 4,
              layer: int = -2,
              to_numpy: bool = False,
              to_float64: bool = False,
              to_cpu: bool = False,
              verbose: bool = True,
              sort: bool = True,
              output_mean_pool: bool = True,
              is_general_sequence: bool = False
              ):

        # Input Has to remove all paddings!!!!!!!!!!
        if self.is_finetune:
            if self.model_name == 'ReformerModelWithLMHead':
                sequences = self._reformer_padding_input(sequences)
            if output_mean_pool:
                return self.embed_inner(sequences=sequences,
                                        batch_size=batch_size,
                                        layer=-1,
                                        verbose=verbose,
                                        output_mean_pool=output_mean_pool,
                                        is_general_sequence=is_general_sequence
                                        )
            else:
                embeddings = self.embed_inner(sequences=sequences,
                                              batch_size=batch_size,
                                              layer=-1,
                                              verbose=verbose,
                                              output_mean_pool=output_mean_pool,
                                              is_general_sequence=is_general_sequence
                                              )
                pad_embeddings, embedding_lengths = self._pad_embeddings(embeddings, False)
                return pad_embeddings, embedding_lengths
        else:
            if sort:
                # sequences = sorted(sequences, key=lambda x: len(x), reverse=True)
                seq_lens = [len(x) for x in sequences]
                sort_indices = np.argsort(seq_lens)[::-1]  # from longest to shortest
                sequences = np.array(sequences)[sort_indices]
                recover_indices = np.argsort(sort_indices)

            with torch.no_grad():
                embeddings = self.embed_inner(sequences=sequences,
                                              batch_size=batch_size,
                                              layer=layer,
                                              verbose=verbose,
                                              output_mean_pool=output_mean_pool,
                                              to_cpu=to_cpu,
                                              is_general_sequence=is_general_sequence
                                              )

            if output_mean_pool:
                embeddings = _embedding_postprocess(embeddings, to_numpy, to_float64, to_cpu)
                assert not torch.isinf(embeddings).any(), ipdb.set_trace()
                assert not torch.isnan(embeddings).any(), ipdb.set_trace()
                if sort:
                    embeddings = embeddings[recover_indices]
                return embeddings
            else:
                # do padding
                pad_embeddings, embedding_lengths = self._pad_embeddings(embeddings, to_cpu)
                pad_embeddings = _embedding_postprocess(pad_embeddings, to_numpy, to_float64, to_cpu)
                if sort:
                    pad_embeddings = pad_embeddings[recover_indices]

                return pad_embeddings, embedding_lengths
