import copy
import collections
from easydict import EasyDict as edict
from .utils import load_save_json


class WhiteSpaceTokenizer():
    def __init__(self, vocab_path=None):
        self.cls_token_id = 0
        self.pad_token_id = 1
        self.sep_token_id = 2
        self.unk_token_id = 3
        self.mask_token_id = 4
        self.added_tokens_encoder = {}
        self.mask_token = '[MASK]'
        self._pad_token = '[PAD]'
        self.base_vocab = {
            '[CLS]': self.cls_token_id,
            '[PAD]': self.pad_token_id,
            '[SEP]': self.sep_token_id,
            '[UNK]': self.unk_token_id,
            '[MASK]': self.mask_token_id,
        }
        self.game_id_cn_char_map = None
        self.use_bpe = False

        if vocab_path:
            self.vocab = {**load_save_json(vocab_path, 'load'), **self.base_vocab}
            print(
                f"[Init White Space Tokenzier] load vocab from path {vocab_path}, total vocab size: {len(self.vocab)}")
        else:
            self.vocab = copy.deepcopy(self.base_vocab)
            print(f"[Init White Space Tokenzier] create new vocab,  total vocab size: {len(self.vocab)}")

        self.word_freq = collections.defaultdict(lambda: 0)

        # temp

        # tokenizer.vocab = {
        #     '[CLS]': 0,
        #     '[PAD]': 1,
        #     '[SEP]': 2,
        #     '[UNK]': 3,
        #     '[MASK]': 4,
        # }

        # temp_word_freq = sorted(tokenizer.word_freq.items(), key=lambda x: x[1], reverse=True)
        # new_word_vocab  = {text_freq[0]:i+5 for i, text_freq in enumerate(temp_word_freq[:50000])}

    @property
    def max_vocab_index(self):
        return max(self.vocab.values())

    def add_vocab_from_list(self, text_list):
        for text in text_list:
            self.word_freq[text] += 1
            if text not in self.vocab:
                next_vocab_index = self.max_vocab_index + 1
                self.vocab[text] = next_vocab_index

    def encode(self, text):
        texts_split = text.split(' ')
        output = edict()
        ids_split = []

        for text in texts_split:
            ids_split.append(self.vocab.get(text, self.vocab['[UNK]']))

        output.ids = ids_split
        output.tokens = texts_split
        return output

    def _is_init_list_consecutive(self, int_list):
        return sorted(int_list) == list(range(min(int_list), max(int_list) + 1))

    def sort_vocab(self, words):
        sorted_vocab = {}
        base_vocab_max_value = max(self.base_vocab.values())
        next_vocab_i = base_vocab_max_value + 1
        for word in words:
            if word in self.base_vocab.keys():
                continue
            else:
                sorted_vocab[word] = next_vocab_i
                next_vocab_i += 1
        assert set(sorted_vocab.values()).intersection(self.base_vocab.values()) == set()
        sorted_vocab = {**sorted_vocab, **self.base_vocab}
        assert self._is_init_list_consecutive(list(sorted_vocab.values()))
        return sorted_vocab

    def squeeze_vocab_by_freq(self, max_vocab_size):
        max_vocab_size_wo_base = max_vocab_size - len(self.base_vocab)
        word_freq = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)
        print(f"Top 10 freq words: {word_freq[:10]}")
        print(f"Last 10 freq words: {word_freq[-10:]}")
        most_freq_words = sorted([x[0] for x in word_freq[:max_vocab_size_wo_base]])
        self.vocab = self.sort_vocab(most_freq_words)
        assert len(self.vocab) <= max_vocab_size
        print(f"Squeeze vocab to size {len(self.vocab)}, max_vocab_size is set to {max_vocab_size}")

    def resort_vocab(self):
        sorted_vocab_keys_wo_base = [x for x, y in self.vocab.items() if x not in self.base_vocab]
        self.vocab = self.sort_vocab(sorted_vocab_keys_wo_base)
        print(f"Resort vocab done! Total: {len(self.vocab)}")

    def get_vocab(self):
        return self.vocab
