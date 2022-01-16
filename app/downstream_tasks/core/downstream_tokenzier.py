import ipdb
import copy
import collections
import numpy as np
from .config import tokenizer_config
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class WhitespaceTokenizer:
    def __init__(self,
                 task,
                 is_debug=False):
        self._base_vocab = {
            '[CLS]': 0,
            '[PAD]': 1,
            '[SEP]': 2,
            '[UNK]': 3,
            '[MASK]': 4

        }
        self._vocab = copy.deepcopy(self.base_vocab)
        self.word_freq = []
        self.unk_percent = tokenizer_config[task].unk_percent_debug if is_debug \
            else tokenizer_config[task].unk_percent
        self.max_vocab_size = tokenizer_config[task].max_vocab_debug if is_debug \
            else tokenizer_config[task].max_vocab

    def create_vocab(self, inputs, verbose=False):
        for input in inputs:
            for x in input:
                if x not in self.base_vocab:
                    self.word_freq.append(x)
        word_count = collections.Counter(self.word_freq).items()
        total_word_count = sum([x[1] for x in word_count if x[0] not in self.base_vocab])
        word_count = sorted(word_count, key=lambda x: x[1], reverse=True)

        acc_word_freq = 0
        for word, word_freq in word_count:
            acc_word_freq += word_freq
            acc_percent = acc_word_freq / total_word_count
            if verbose:
                print(f"acc_percent: {acc_percent}")
            if acc_percent > self.unk_percent:
                break
            else:
                self._vocab[word] = self.max_vocab_index + 1
                if len(self.vocab) >= self.max_vocab_size:
                    break
        print(f"Create vocab done, total vocab: {len(self.vocab)}, unk_precent: {self.unk_percent}")

    def tokenize_behaviour_seq(self, input: Union[np.ndarray, List, Tuple]):
        log_design_ids = [x for x_i, x in enumerate(input) if (x_i + 1) % 3 != 0]
        timestamps = input[2::3]
        tokenized_ids = [self.vocab.get(x, self.vocab['[UNK]']) for x in log_design_ids]
        no_pad_len = len([x for x in tokenized_ids if x != self.vocab['[PAD]']])
        return tokenized_ids, no_pad_len, timestamps

    def tokenize_map_ids(self, input: Union[np.ndarray, List, Tuple]):
        tokenized_ids = [self.vocab.get(x, self.vocab['[UNK]']) for x in input]
        no_pad_len = len([x for x in tokenized_ids if x != self.vocab['[PAD]']])
        return tokenized_ids, no_pad_len

    @property
    def base_vocab(self):
        return self._base_vocab

    @property
    def vocab_size(self):
        return len(self._vocab)

    @property
    def padding_idx(self):
        return self._vocab['[PAD]']

    @property
    def vocab(self):
        return self._vocab

    @property
    def max_vocab_index(self):
        return max(self._vocab.values())
