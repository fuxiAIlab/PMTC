import os
import ipdb
import json
import hashlib
import torch
import collections
from tqdm import tqdm
from tokenizers import Tokenizer


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


def _bpe_bow_md5_hash_sample(sample: list):
    assert isinstance(sample, list)
    hash_str = str(sample)
    return hashlib.md5(hash_str.encode('utf-8')).hexdigest()


class BpeBowEmbedder:
    def __init__(self, bpe_tokenizer_path, cn_char_id_mapping_path):
        self.model_name = 'bpe_bow'
        self.pretrain_task = 'bpe_bow'
        self.max_sequence_length = 512  # TODO, 搞清楚这个到底怎么用
        self.embedding_cache = {}
        self.use_token_weights = False

        # Init Bpe tokenizer
        self.cn_char_id_mapping = load_save_json(cn_char_id_mapping_path, 'load')
        self.tokenizer = Tokenizer.from_file(bpe_tokenizer_path)
        self.cn_char_vocab = self.tokenizer.get_vocab()
        self.embedding_dim = len(self.tokenizer.get_vocab())
        self.bow_dim = self.embedding_dim
        self.meta_config = {}

        print(
            f"[BpeBowEmbedder] load bpe tokenizer from {bpe_tokenizer_path}, "
            f"vocab_size: {self.embedding_dim},"
            f" mapping_size: {len(self.cn_char_id_mapping)}")

    def embed(self, samples, **kwargs):
        verbose = kwargs.get('verbose', True)
        bow_tensors = []
        for sample in tqdm(samples, total=len(samples), disable=True if not verbose else False):
            sample = [x for i, x in enumerate(sample) if (i + 1) % 3 != 0]
            sample_md5 = _bpe_bow_md5_hash_sample(sample)
            if sample_md5 not in self.embedding_cache:
                # convert sample to cn_char
                sample_cn_char = [self.cn_char_id_mapping.get(x, self.cn_char_id_mapping['[UNK]']) for x in sample]
                sample_cn_char = ''.join(sample_cn_char)
                tokenized_indices = collections.Counter(self.tokenizer.encode(sample_cn_char).ids)
                bow_tensor = torch.zeros((self.bow_dim))
                for index, count in tokenized_indices.items():
                    bow_tensor[index] = count

                # save to embedding cache
                self.embedding_cache[sample_md5] = bow_tensor
            else:
                bow_tensor = self.embedding_cache[sample_md5]
            bow_tensors.append(bow_tensor)
        bow_tensors = torch.stack(bow_tensors)
        if verbose:
            print(f"Load all bow embedding done!")
        return bow_tensors

    def update_embedding_cache(self, x1, x2):
        pass

    def save_embedding_cache(self):
        pass
