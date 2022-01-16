from tqdm import tqdm
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union
import ipdb
import collections
import random
import torch
from copy import deepcopy
from torch.nn.utils.rnn import pad_sequence

from transformers.tokenization_utils_base import BatchEncoding


def _sample_by_model_predict_prob(prob_tensor, labels):
    dynamic_mask_predict_prob = prob_tensor.clone()
    batch_mask_predict_prob = dynamic_mask_predict_prob.expand(labels.shape[0],
                                                               dynamic_mask_predict_prob.shape[0]).to(labels.device)
    probability_matrix = batch_mask_predict_prob.gather(1, labels)
    avg_model_predict_prob = float(torch.mean(probability_matrix))
    probability_matrix = 1 - probability_matrix
    return probability_matrix, avg_model_predict_prob


def _sample_by_tfidf(idf_tensor, labels):
    token_tf_idfs = []
    for label_i, label in enumerate(labels):
        token_count = torch.bincount(label)  # ->  shape: (max_index_of_label(e.g. 49000), )
        token_freq_pad = torch.zeros_like(idf_tensor)
        token_freq_pad[:len(token_count)] = token_count
        token_tf_idfs.append(token_freq_pad)

    token_tfs = torch.stack(token_tf_idfs)
    token_inver_tfs = 1 / token_tfs
    token_tf_idfs = token_inver_tfs * idf_tensor  # compute the tf-idf value
    probability_matrix = token_tf_idfs.gather(1, labels)
    return probability_matrix


def _tf_idf_decay_func(tf_idf_init_prob, step_i, decay=0.998):
    if step_i == 0:
        return tf_idf_init_prob
    else:
        return tf_idf_init_prob * decay ** step_i


@dataclass
class BpeDataCollatorForLanguageModeling:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    # use_time_embed = False
    def __init__(self,
                 tokenizer,
                 use_time_embed,
                 mlm_probability,
                 pretrain_task,
                 clm_sample_n,
                 use_random_mlm_probability=None,
                 mlm_prob_min=None,
                 mlm_prob_max=None,
                 mask_type='normal',
                 mask_softmax_t=0.5,
                 masker_recorder=None,
                 tf_idf_warmup_decay=None,
                 is_record_mask_ratio=False,
                 return_timestamps=True,
                 softmax_t_decay_mode=None,
                 part_prob_percent=None,
                 part_prob_range=None
                 ):
        self.tokenizer = tokenizer
        self.use_time_embed = use_time_embed
        self.mlm_probability = mlm_probability
        self.pretrain_task = pretrain_task
        self.clm_sample_n = clm_sample_n
        self.CLM_MIN_LEN = 32
        self.use_random_mlm_probability = use_random_mlm_probability
        self.mlm_prob_min = mlm_prob_min
        self.mlm_prob_max = mlm_prob_max
        self.mask_type = mask_type
        self.masker_recorder = masker_recorder
        self._mask_softmax_t = mask_softmax_t
        self.tf_idf_warmup_decay = tf_idf_warmup_decay
        self.is_record_mask_ratio = is_record_mask_ratio
        self.return_timestamps = return_timestamps
        self.softmax_t_decay_mode = softmax_t_decay_mode
        self.softmax_t_range = (0.0001, 0.8)
        self.total_training_step = 0
        self.part_prob_percent = part_prob_percent
        self.part_prob_range = part_prob_range

        if self.mask_type == 'part_prob_linear_increase':
            pmin, pmax = self.part_prob_range
            assert pmin < pmax
            self.part_prob_percent = pmin

        if self.use_random_mlm_probability is not None:
            assert self.mlm_prob_min
            assert self.mlm_prob_max

    @property
    def mask_softmax_t(self):
        return max(self._mask_softmax_t, 1e-10)

    def _linear_decay_t(self, t_min, t_max, total_step, step):
        return (t_min - t_max) / total_step * step + t_max

    def adjust_part_prob_percent(self, step_now):
        if self.mask_type == 'part_prob_linear_increase':
            p_min, p_max = self.part_prob_range
            self.part_prob_percent = p_max - self._linear_decay_t(p_min, p_max, self.total_training_step, step_now)

    def adjust_mask_softmax_t(self, step_now):
        t_min, t_max = self.softmax_t_range
        if self.softmax_t_decay_mode in {'linear'}:
            self._mask_softmax_t = self._linear_decay_t(t_min, t_max, self.total_training_step, step_now)
        # 凹
        elif self.softmax_t_decay_mode == 'exponential_concave':
            tau = 0.2
            linear_t = self._linear_decay_t(t_min, t_max, self.total_training_step, step_now)
            self._mask_softmax_t = - np.exp(-linear_t / tau) + t_max
        # 凸
        elif self.softmax_t_decay_mode == 'exponential_convex':
            tau = 0.2
            linear_t = self._linear_decay_t(t_min, t_max, self.total_training_step, step_now)
            self._mask_softmax_t = np.exp(-(1 - linear_t) / tau)
        elif self.softmax_t_decay_mode == 'by_prob':
            self._mask_softmax_t = self.masker_recorder.mean_prob_tensor
            # print(f"Set mask softmax t to {self._mask_softmax_t}")
        else:
            return None

    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            examples = [e["input_ids"] for e in examples]
        batch_ids, batch_timestamps = self._tensorize_batch(examples)
        if self.pretrain_task == 'mlm':
            inputs, labels, batch_timestamps, attention_mask = self.mlm_mask_tokens(batch_ids, batch_timestamps)
        elif self.pretrain_task == 'clm':
            inputs, labels, batch_timestamps, attention_mask = self.clm_mask_tokens(batch_ids, batch_timestamps)
        else:
            raise NotImplementedError
        if self.return_timestamps:
            return {"input_ids": inputs,
                    "labels": labels,
                    'timestamps': batch_timestamps,
                    'attention_mask': attention_mask}
        else:
            return {"input_ids": inputs,
                    "labels": labels,
                    'attention_mask': attention_mask}

    def _tensorize_batch(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ):
        batch_ids = []
        batch_timestamps = []
        for cn_char_subword, timestamp_subword in examples:
            batch_ids.append(cn_char_subword)
            batch_timestamps.append(timestamp_subword)
        batch_ids = torch.stack(batch_ids)
        return batch_ids, batch_timestamps
        #
        # # In order to accept both lists of lists and lists of Tensors
        # if isinstance(examples[0], (list, tuple)):
        #     examples = [torch.tensor(e, dtype=torch.long) for e in examples]
        # length_of_first = examples[0].size(0)
        # are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        # if are_tensors_same_length:
        #     return torch.stack(examples, dim=0)
        # else:
        #     if self.tokenizer._pad_token is None:
        #         raise ValueError(
        #             "You are attempting to pad samples but the tokenizer you are using"
        #             f" ({self.tokenizer.__class__.__name__}) does not have one."
        #         )
        #     return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def _compute_pad_len(self, labels):
        pad_lens = []
        for label in labels:
            non_pad_length = len(label[label != self.tokenizer.pad_token_id])
            pad_lens.append(non_pad_length)
        pad_lens = torch.tensor(pad_lens).unsqueeze(1)
        return pad_lens

    def _handle_prob_overshoot(self,
                               probability_matrix,
                               pad_lens,
                               overshoot_threshold=1.0):
        is_exist_overshoot_indices = bool((probability_matrix > overshoot_threshold).any())
        if is_exist_overshoot_indices:
            for seq_i, seq_prob in enumerate(probability_matrix):
                gt_1_mask = seq_prob > overshoot_threshold
                if bool(gt_1_mask.any()):
                    overshoot_value = int(seq_prob[gt_1_mask])
                    distribute_value = float((overshoot_value - overshoot_threshold)) / float(pad_lens[seq_i] - 1)
                    seq_prob[~gt_1_mask] = seq_prob[~gt_1_mask] + distribute_value
                    seq_prob[gt_1_mask] = overshoot_value
            return True
        else:
            return False

    def mlm_mask_tokens(self, inputs: torch.Tensor, batch_timestamps, verbose=False):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.use_random_mlm_probability:
            mlm_probability = random.uniform(self.mlm_prob_min, self.mlm_prob_max)
        else:
            mlm_probability = self.mlm_probability
        self.mlm_probability = mlm_probability

        assert 0 < mlm_probability < 1

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        if self.mask_type.startswith('part_prob'):
            assert self.part_prob_percent is not None
            random_value = random.random()
            if random_value < self.part_prob_percent:
                mask_type = 'posterior_prob'
            else:
                mask_type = 'normal'
            # print(
            #     f"random_value: {random_value}, set mask_type to {mask_type}, part_prob_value: {self.part_prob_percent}")
        else:
            mask_type = self.mask_type

        if mask_type in {'posterior_prob', 'tf_idf', 'posterior_prob_with_tf_idf_warmup'}:
            if mask_type == 'posterior_prob':
                probability_matrix, avg_model_predict_prob = _sample_by_model_predict_prob(
                    self.masker_recorder.prob_tensor, labels)
                print(f"Model avg predict prob: {avg_model_predict_prob}, t: {self.mask_softmax_t}")
            elif mask_type == 'tf_idf':
                probability_matrix = _sample_by_tfidf(self.masker_recorder.idf_tensor, labels)
            elif mask_type == 'posterior_prob_with_tf_idf_warmup':
                tf_idf_prob = _tf_idf_decay_func(1.0,
                                                 self.masker_recorder.train_step,
                                                 decay=self.tf_idf_warmup_decay)
                self.masker_recorder.tf_idf_warm_up_probs.append(tf_idf_prob)
                random_prob = random.random()
                if verbose:
                    print(f"[Warm up by tfidf] step i: {self.masker_recorder.train_step},"
                          f" tf_idf_prob: {tf_idf_prob},"
                          f" random_prob: {random_prob}")
                if random_prob < tf_idf_prob:
                    probability_matrix = _sample_by_tfidf(self.masker_recorder.idf_tensor, labels)
                else:
                    probability_matrix, avg_model_predict_prob = _sample_by_model_predict_prob(
                        self.masker_recorder.prob_tensor, labels)
            else:
                raise NotImplementedError

            # temp compute the freq of tokens in each sample
            pad_token_indices = torch.where(labels == self.tokenizer.pad_token_id)
            # TODO, 这里获取NO PAD长度的写的不太好，但是目前也想不出不用for llop的办法
            pad_lens = self._compute_pad_len(labels)
            probability_matrix[pad_token_indices] = float('-inf')
            probability_matrix = torch.softmax(probability_matrix / self.mask_softmax_t, dim=1)
            probability_matrix = probability_matrix * mlm_probability * pad_lens
            # is_overshoot = self._handle_prob_overshoot(probability_matrix,
            #                                            pad_lens,
            #                                            overshoot_threshold=1.0)

            is_overshoot = False

            if is_overshoot:
                self.masker_recorder.overshoot_count += 1

            probability_matrix[probability_matrix >= 1.0] = 1.0
            probability_matrix[probability_matrix <= 0.0] = 0.0
            # ----------------------------------------------------------------------------------------------------------
            # Print for debug
            # ----------------------------------------------------------------------------------------------------------
            if verbose:
                non_pad_token_indices = torch.where(labels != self.tokenizer.pad_token_id)
                print(f"[Probability Matrix] min-{torch.min(probability_matrix[non_pad_token_indices])},"
                      f"max-{torch.max(probability_matrix[non_pad_token_indices])},"
                      f"avg-{torch.mean(probability_matrix[non_pad_token_indices])},"
                      f"softmax_t: {self.mask_softmax_t}")
                print_masked_indices = torch.bernoulli(probability_matrix).bool()
                for pad_len_i, pad_len in enumerate(pad_lens):
                    print(
                        f"[Mask ratio-{pad_len_i}]: "
                        f"{collections.Counter(print_masked_indices[pad_len_i].tolist())[True] / int(pad_len)}")
            # ----------------------------------------------------------------------------------------------------------
            try:
                masked_indices = torch.bernoulli(probability_matrix).bool()
            except:
                ipdb.set_trace()

        elif mask_type == 'lowest_prob':

            # # ----------------------------------------------------------------------------------------------------------
            # # OLD version
            # # ----------------------------------------------------------------------------------------------------------
            # RANDOM_RATIO = 0.0
            # dynamic_mask_predict_prob = self.masker_recorder.prob_tensor.clone()
            # # device = dynamic_mask_predict_prob.to(dynamic_mask_predict_prob.device)
            # probability_matrix = torch.zeros(labels.shape)
            # seq_len = probability_matrix.shape[1]
            # pad_start_indices = []
            # for label_i, label in enumerate(labels):
            #     padding_indices = torch.where(label == self.tokenizer.pad_token_id)[0]
            #     if padding_indices.shape[0] == 0:
            #         pad_start_index = len(label)
            #     else:
            #         pad_start_index = int(padding_indices[0])
            #     pad_start_indices.append(pad_start_index)
            #     probability_matrix[label_i] = dynamic_mask_predict_prob[label]
            #     probability_matrix[label_i][padding_indices] = float('inf')
            #
            #     # label_prob = dynamic_mask_predict_prob[label]
            #     # label_prob[padding_indices] = float('inf')
            #     # ipdb.set_trace()
            #     # label_prob = (1 - RANDOM_RATIO) * label_prob + RANDOM_RATIO * torch.rand((len(label_prob, ))).to(
            #     #     label_prob.device)
            #     # top_percent_label_indices = torch.argsort(label_prob)[:int(len(label_prob) * self.mlm_probability)]
            #     # masked_index = torch.zeros_like(label_prob, dtype=int)
            #     # masked_index[top_percent_label_indices] =
            #     # masked_index = masked_index.bool()
            #     # masked_indices.append(masked_index)
            # # masked_indices = torch.stack(masked_indices)  # batch_size x max_seq_length
            #
            # probability_matrix = (1 - RANDOM_RATIO) * probability_matrix + RANDOM_RATIO * torch.rand_like(
            #     probability_matrix)
            # top_percent_label_indices = torch.argsort(probability_matrix)[:, :int(seq_len * self.mlm_probability)]
            # masked_indices = torch.zeros_like(probability_matrix, dtype=int)
            # for masked_index_i, masked_index in enumerate(masked_indices):
            #     top_percent_label_index = top_percent_label_indices[masked_index_i]
            #     top_percent_label_index = top_percent_label_index[
            #         top_percent_label_index < pad_start_indices[masked_index_i]]
            #     masked_index[top_percent_label_index] = 1
            # masked_indices = masked_indices.bool()
            # # ----------------------------------------------------------------------------------------------------------

            # ----------------------------------------------------------------------------------------------------------
            # NEW version
            # ----------------------------------------------------------------------------------------------------------
            RANDOM_RATIO = 1e-6  # 1e-6
            dynamic_mask_predict_prob = self.masker_recorder.prob_tensor.clone()
            batch_mask_predict_prob = dynamic_mask_predict_prob.expand(labels.shape[0],
                                                                       dynamic_mask_predict_prob.shape[0]).to(
                labels.device)
            probability_matrix = batch_mask_predict_prob.gather(1, labels)
            # seq_len = probability_matrix.shape[1]
            pad_token_indices = torch.where(labels == self.tokenizer.pad_token_id)
            probability_matrix[pad_token_indices] = float('inf')  # batch_size x max_seq_len
            probability_matrix = (1 - RANDOM_RATIO) * probability_matrix + RANDOM_RATIO * torch.rand_like(
                probability_matrix)
            pad_lens = self._compute_pad_len(labels)
            top_percent_label_indices = []
            argsort_probability_matrix = torch.argsort(probability_matrix)
            for pad_len_i, pad_len in enumerate(pad_lens):
                top_percent_label_indices.append(
                    argsort_probability_matrix[pad_len_i][:int(pad_len * self.mlm_probability)])

            # top_percent_label_indices = torch.argsort(probability_matrix)[:, :int(seq_len * self.mlm_probability)]
            temp_indices = torch.cat([torch.full(x.shape, i) for i, x in enumerate(top_percent_label_indices)]).long()
            top_percent_label_fancy_index = (temp_indices, torch.cat(top_percent_label_indices))
            masked_indices = torch.zeros_like(probability_matrix, dtype=int)
            masked_indices[top_percent_label_fancy_index] = 1
            masked_indices[pad_token_indices] = 0
            masked_indices = masked_indices.bool()
            # ----------------------------------------------------------------------------------------------------------

            # # ----------------------------------------------------------------------------------------------------------
            # # Compute softmax version & compare
            # # ----------------------------------------------------------------------------------------------------------
            # mask_softmax_t = 0.00001
            # probability_matrix_softmax = batch_mask_predict_prob.gather(1, labels)
            # probability_matrix_softmax = torch.softmax(probability_matrix_softmax / mask_softmax_t, dim=1)
            # probability_matrix_softmax = probability_matrix_softmax * mlm_probability * pad_lens
            # masked_indices_softmax = torch.bernoulli(probability_matrix_softmax).bool()
            #
            # for pad_len_i, pad_len in enumerate(pad_lens):
            #     print("-" * 78)
            #     print(
            #         f"[lowest_prob][Mask ratio-{pad_len_i}]: "
            #         f"{collections.Counter(masked_indices[pad_len_i].tolist())[True] / int(pad_len)}")
            #     print(
            #         f"[Softmax][Mask Mask-{pad_len_i}]: "
            #         f"{collections.Counter(masked_indices_softmax[pad_len_i].tolist())[True] / int(pad_len)}")
            # # ----------------------------------------------------------------------------------------------------------

        elif mask_type == 'normal':
            # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
            probability_matrix = torch.full(labels.shape, mlm_probability)
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
            if self.tokenizer._pad_token is not None:
                padding_mask = labels.eq(self.tokenizer.pad_token_id)
                probability_matrix.masked_fill_(padding_mask, value=0.0)
            pad_lens = None
            masked_indices = torch.bernoulli(probability_matrix).bool()  # batch_size x max_seq_length
        else:
            raise NotImplementedError

        if self.masker_recorder is not None:
            if self.masker_recorder.record_snapshot:
                self.masker_recorder.step_mask_probabilities.extend(
                    probability_matrix[probability_matrix > 0.0].tolist())
                if self.mask_type == 'part_prob' and mask_type == 'posterior_prob':
                    keep_N = 16
                    record_point_N = 200  # For debug
                    max_sample_per_step = 32

                    if self.masker_recorder.train_step % max(int(self.masker_recorder.total_steps / record_point_N),
                                                             10) == 0:
                        train_step_counts = collections.Counter(
                            self.masker_recorder.step_sample_mask_distributions['train_step'])
                        current_step_count = train_step_counts[self.masker_recorder.train_step]
                        if current_step_count >= max_sample_per_step:
                            pass
                        else:
                            for label, mask_prob in zip(labels[:keep_N], probability_matrix[:keep_N]):
                                label = label[label != 1]
                                mask_prob = mask_prob[:len(label)]
                                self.masker_recorder.step_sample_mask_distributions['train_step'].append(
                                    self.masker_recorder.train_step)
                                self.masker_recorder.step_sample_mask_distributions['tokens'].append(
                                    tuple(label.tolist()))
                                self.masker_recorder.step_sample_mask_distributions['mask_prob'].append(
                                    tuple(mask_prob.tolist()))
                                self.masker_recorder.step_sample_mask_distributions['softmax_t'].append(
                                    self.mask_softmax_t)
                                self.masker_recorder.step_sample_mask_distributions['avg_model_prob'].append(
                                    max(avg_model_predict_prob, 1e-10))
        # # save label maskes
        # self.masker_recorder

        if self.is_record_mask_ratio:
            if pad_lens is None:
                pad_lens = self._compute_pad_len(labels)
            for masked_index_i, masked_index in enumerate(masked_indices):
                mask_ratio = collections.Counter(masked_index.tolist())[True] / int(pad_lens[masked_index_i])
                self.masker_recorder.mask_ratios.append(mask_ratio)

        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.tokenizer.max_len, labels.shape, dtype=torch.long)

        inputs[indices_random] = random_words[indices_random]

        # Compute Attention Mask
        # ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

        attention_mask = torch.ones_like(inputs)
        attention_mask[inputs.eq(self.tokenizer.pad_token_id)] = 0

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels, batch_timestamps, attention_mask

    def clm_mask_tokens(self, inputs: torch.Tensor, batch_timestamps):
        """

        Parameters
        ----------
        inputs: tensor, shape: batch_size x max_seq_len
            example:
                tensor([[15086,  8773, 10116,  ...,     1,     1,     1],
                        [13689,  1683,  1613,  ...,     1,     1,     1]])

        batch_timestamps: List

        Returns
        -------

        """
        inputs_clone = inputs.clone()
        input_lens = [len(x[x != 1]) for x in inputs]

        new_inputs = []
        new_labels = []
        attention_masks = []

        for i, input_len in enumerate(input_lens):

            if input_len <= self.CLM_MIN_LEN:
                continue

            clm_samples = inputs_clone[i].repeat(self.clm_sample_n, 1)
            sample_pad_mask = torch.ones(self.clm_sample_n, inputs_clone.shape[1]).bool()
            sample_pad_mask_view = sample_pad_mask.view(-1)

            # 最小是当前样本长度的1/4或者是32
            clm_sample_len = random.sample(range(self.CLM_MIN_LEN, input_len), self.clm_sample_n)
            unmask_view_indices = []
            for i, x in enumerate(clm_sample_len):
                unmask_view_indices.extend(list(range(i * sample_pad_mask.shape[1], i * sample_pad_mask.shape[1] + x)))

            sample_pad_mask_view[unmask_view_indices] = False
            clm_samples[sample_pad_mask] = self.tokenizer.pad_token_id

            # set labels
            cls_labels = clm_samples.clone()
            cls_label_mask = torch.zeros_like(cls_labels).bool()
            cls_label_mask = ~torch.scatter(cls_label_mask, 1, (torch.tensor(clm_sample_len) - 1).unsqueeze(1), True)
            cls_labels[cls_label_mask] = -100
            clm_samples[~cls_label_mask] = self.tokenizer.pad_token_id

            # set attentions
            attention_pad_mask = sample_pad_mask.clone()
            attention_pad_mask = ~torch.scatter(attention_pad_mask, 1, (torch.tensor(clm_sample_len) - 1).unsqueeze(1),
                                                True)
            attention_pad_mask = attention_pad_mask.long()

            # some assertions
            temp_assert_index = len(clm_samples[0][clm_samples[0] != 1])
            assert cls_labels[0][temp_assert_index] != -100
            assert clm_samples[0][temp_assert_index] == self.tokenizer.pad_token_id
            assert clm_samples[0][temp_assert_index - 1] != self.tokenizer.pad_token_id, ipdb.set_trace()
            assert attention_pad_mask[0][temp_assert_index] == 0  # Mask for label position
            assert attention_pad_mask[0][temp_assert_index - 1] == 1  # Unmask for previous position

            new_inputs.append(clm_samples)
            new_labels.append(cls_labels)
            attention_masks.append(attention_pad_mask)

        new_inputs = torch.cat(new_inputs)
        new_labels = torch.cat(new_labels)
        attention_masks = torch.cat(attention_masks)

        return new_inputs, new_labels, batch_timestamps, attention_masks
