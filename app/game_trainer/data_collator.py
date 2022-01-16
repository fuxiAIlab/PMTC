from dataclasses import dataclass
from typing import Dict, List, Tuple, Union
import ipdb
import torch
from copy import deepcopy
from torch.nn.utils.rnn import pad_sequence

from transformers.tokenization_utils_base import BatchEncoding


@dataclass
class DataCollatorForGameMultiTask:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    mlm: bool = True

    def __init__(self,
                 behave_tokenizer=None,
                 design_tokenizer=None,
                 is_task0=True,
                 is_task1=True,
                 is_task2=True,
                 is_task3=True,
                 is_task4=True,
                 task0_mlm_prob=0.15,
                 task1_mlm_prob=0.15,
                 task2_mlm_prob=0.15,
                 task3_mlm_prob=0.15,
                 task4_mlm_prob=0.15
                 ):
        self.is_task0 = is_task0
        self.is_task1 = is_task1
        self.is_task2 = is_task2
        self.is_task3 = is_task3
        self.is_task4 = is_task4
        self.behave_tokenizer = behave_tokenizer
        self.design_tokenizer = design_tokenizer
        self.task0_mlm_prob = task0_mlm_prob
        self.task1_mlm_prob = task1_mlm_prob
        self.task2_mlm_prob = task2_mlm_prob
        self.task3_mlm_prob = task3_mlm_prob
        self.task4_mlm_prob = task4_mlm_prob

    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ):
        if isinstance(examples[0], (dict, BatchEncoding)):
            examples = [e["input_ids"] for e in examples]
        batch = self._tensorize_batch(examples)
        return self.mask_tokens(batch)

    def _tensorize_batch(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> torch.Tensor:
        # In order to accept both lists of lists and lists of Tensors
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def mask_input_tokens(self, inputs, target_indices, indices_replaced):
        target_input = inputs[:, target_indices]
        target_input[indices_replaced] = self.behave_tokenizer.convert_tokens_to_ids(
            self.behave_tokenizer.mask_token)
        inputs[:, target_indices] = target_input
        return inputs

    def replace_input_tokens_random(self, inputs, target_indices, indices_random, labels):
        random_words = torch.randint(self.behave_tokenizer.max_len, labels.shape, dtype=torch.long)
        target_input = inputs[:, target_indices]
        target_input[indices_random] = random_words[indices_random]
        inputs[:, target_indices] = target_input
        return inputs

    def mask_for_task0(self, inputs, labels, target_indices, mask_prob=None):
        if mask_prob is None:
            mask_prob = self.task0_mlm_prob

        assert self.behave_tokenizer is not None
        labels = labels[:, target_indices]

        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, mask_prob)
        special_tokens_mask = [
            self.behave_tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
            labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.behave_tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.behave_tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs = self.mask_input_tokens(inputs, target_indices, indices_replaced)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        inputs = self.replace_input_tokens_random(inputs, target_indices, indices_random, labels)

        return inputs, labels

    def mask_for_timestamps(self, inputs, labels, target_indices, mask_prob):
        labels = labels[:, target_indices].float()

        # normalize by bach mean & variances
        labels = (labels - torch.mean(labels)) / torch.std(labels)

        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, mask_prob)
        special_tokens_mask = [
            self.behave_tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
            labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.behave_tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.behave_tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        target_inputs = inputs[:, target_indices]
        target_inputs[masked_indices] = -1
        inputs[:, target_indices] = target_inputs
        return inputs, labels

    def mask_for_task1(self, inputs, labels, behave_id_indices, design_id_indices, timestamp_indices):
        inputs, labels = self.mask_for_task0(inputs, labels, behave_id_indices, mask_prob=self.task1_mlm_prob)

        # MASK all design IDs
        inputs[:, design_id_indices] = self.design_tokenizer.convert_tokens_to_ids(self.design_tokenizer.mask_token)

        # MASK all timestamps to -1
        inputs[:, timestamp_indices] = -1

        return inputs, labels

    def mask_for_task3(self, inputs, labels, behave_id_indices, design_id_indices, timestamp_indices):
        inputs, labels = self.mask_for_timestamps(inputs, labels, timestamp_indices, mask_prob=self.task3_mlm_prob)
        inputs[:, design_id_indices] = self.design_tokenizer.convert_tokens_to_ids(self.design_tokenizer.mask_token)
        inputs[:, behave_id_indices] = self.behave_tokenizer.convert_tokens_to_ids(self.behave_tokenizer.mask_token)
        return inputs, labels

    def mask_for_task4(self, inputs, labels, behave_id_indices, design_id_indices, timestamp_indices):
        inputs, labels = self.mask_for_task0(inputs, labels, design_id_indices, mask_prob=self.task4_mlm_prob)

        # MASK all behave IDs
        inputs[:, behave_id_indices] = self.behave_tokenizer.convert_tokens_to_ids(self.behave_tokenizer.mask_token)

        # MASK all timestamps to -1
        inputs[:, timestamp_indices] = -1
        return inputs, labels

    def mask_for_task2(self, inputs, labels, design_id_indices, timestamp_indices):
        assert self.design_tokenizer is not None

        design_labels = labels[:, design_id_indices]

        # normalize timestamp
        timestamps = labels[:, timestamp_indices].float()
        timestamps = (timestamps - torch.mean(timestamps)) / torch.std(timestamps)

        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(design_labels.shape, self.task2_mlm_prob)
        special_tokens_mask = [
            self.design_tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
            design_labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.design_tokenizer._pad_token is not None:
            padding_mask = design_labels.eq(self.design_tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # choose some part of design id for prediction, mask all those positions
        design_labels[~masked_indices] = -100  # We only compute loss on masked tokens
        target_design_inputs = inputs[:, design_id_indices]
        target_design_inputs[masked_indices] = self.design_tokenizer.convert_tokens_to_ids(
            self.design_tokenizer.mask_token)
        inputs[:, design_id_indices] = target_design_inputs

        # mask for timestamps as well, same position as design_id
        timestamps[~masked_indices] = -100
        target_timestamp_inputs = inputs[:, timestamp_indices]
        target_timestamp_inputs[masked_indices] = -1
        inputs[:, timestamp_indices] = target_timestamp_inputs

        return inputs, (design_labels, timestamps)

    def mask_tokens(self, inputs: torch.Tensor):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        cut_index1 = int(labels.shape[1] / 3)
        cut_index2 = cut_index1 * 2

        behave_id_indices = list(range(0, cut_index1))
        design_id_indices = list(range(cut_index1, cut_index2))
        timestamp_indices = list(range(cut_index2, inputs.shape[1]))

        task_result = {}

        if self.is_task0:
            task0_inputs, task0_labels = self.mask_for_task0(inputs.clone(),
                                                             labels.clone(),
                                                             deepcopy(behave_id_indices))
            task_result['task0'] = (task0_inputs, task0_labels)
        if self.is_task1:
            task1_inputs, task1_labels = self.mask_for_task1(inputs.clone(),
                                                             labels.clone(),
                                                             deepcopy(behave_id_indices),
                                                             deepcopy(design_id_indices),
                                                             deepcopy(timestamp_indices))
            task_result['task1'] = (task1_inputs, task1_labels)
        if self.is_task2:
            task2_inputs, task2_labels = self.mask_for_task2(inputs.clone(),
                                                             labels.clone(),
                                                             deepcopy(design_id_indices),
                                                             deepcopy(timestamp_indices))
            task_result['task2'] = (task2_inputs, task2_labels)
        if self.is_task3:
            task3_inputs, task3_labels = self.mask_for_task3(inputs.clone(),
                                                             labels.clone(),
                                                             deepcopy(behave_id_indices),
                                                             deepcopy(design_id_indices),
                                                             deepcopy(timestamp_indices)
                                                             )
            task_result['task3'] = (task3_inputs, task3_labels)
        if self.is_task4:
            task4_inputs, task4_labels = self.mask_for_task4(inputs.clone(),
                                                             labels.clone(),
                                                             deepcopy(behave_id_indices),
                                                             deepcopy(design_id_indices),
                                                             deepcopy(timestamp_indices)
                                                             )
            task_result['task4'] = (task4_inputs, task4_labels)

        return task_result
