"""
This is a compact training script for game bert
"""
import collections
import logging
import math
import numpy as np
import os
import ipdb
import hashlib
import torch
import pickle
import time
import sys

from multiprocessing import Lock

sys.path.append('..')
import datetime

from tqdm import tqdm
from torch.utils.data.dataset import Dataset

from typing import Optional
from args_utils import DataTrainingArguments

# Init logger
logger = logging.getLogger(__name__)

from transformers import (
    PreTrainedTokenizer,
)


def convert_abs_time_to_relative(time_gap_block):
    time_gap_block = np.array(time_gap_block).astype(int)
    # 这里做一下转换，一天有86400秒
    time_gap0 = datetime.datetime.fromtimestamp(time_gap_block[0])
    today_start = datetime.datetime(year=time_gap0.year, month=time_gap0.month, day=time_gap0.day)
    today_start_timestamp = int(time.mktime(today_start.timetuple()))
    time_gaps = time_gap_block - today_start_timestamp
    return time_gaps


def md5_time_embedidng(sample: list,
                       config_dict: dict):
    assert isinstance(sample, list)
    hash_str = str(sample) + str(sorted(config_dict.items()))
    return hashlib.md5(hash_str.encode('utf-8')).hexdigest()


class BpeTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
            self,
            hdf_data,
            tokenizer,
            hdf5_file_path: str,
            block_size: int,
            use_time_embed: bool = False,
            debugN=None,
            is_timestamp_rnn=False,
            cache_dir=None
    ):
        self.tokenizer = tokenizer
        self.use_time_embed = use_time_embed
        assert os.path.isfile(hdf5_file_path), f"Input file path {hdf5_file_path} not found"
        self.examples = []
        self.time_gaps = []
        self.design_ids = []
        self.pickle_lock = Lock()
        self.temp_samples = []
        self.save_temp_samples = False

        self.cache_dir = cache_dir
        if use_time_embed:
            with self.pickle_lock:
                self.cache_save_path = os.path.join(cache_dir, 'bpe_time_embedding_cahce.pickle')
                if os.path.isfile(self.cache_save_path):
                    self.time_embedding_cache = pickle.load(open(self.cache_save_path, 'rb'))
                    print(f"[Load time embedding cache] load from {self.cache_save_path} SUCCESS")
                else:
                    self.time_embedding_cache = {}
                    print("[Load time embedding cache] Create new time embedding cache")

        for sample_i, sample in tqdm(enumerate(hdf_data), total=len(hdf_data)):

            if debugN:
                if sample_i >= debugN:
                    print(f"[DEBUG N] Stop loading data, debug N is set to {debugN}")
                    break
            sample = [x for x in sample if x != '[PAD]']
            text_block = ' '.join(sample)
            # --------------------------------------------------------------------------------------------------
            # CODE BY PJS
            # --------------------------------------------------------------------------------------------------

            # Normalize text block
            if use_time_embed:
                pure_text_block = [x for i, x in enumerate(text_block.split(' ')) if (i + 1) % 3 != 0]
                time_gap_block = text_block.split(' ')[2::3]

                # This is only for data assertion
                temp_test_time_gap = time_gap_block[0]
                temp_date_obj = datetime.datetime.fromtimestamp(int(temp_test_time_gap))

                assert 2019 < temp_date_obj.year < 2022
                time_gap_block = [y for x in zip(time_gap_block, time_gap_block) for y in x]
                text_block = ''.join(pure_text_block)

                assert len(pure_text_block) == len(time_gap_block)

                # Create dict for time/cn_char mapping
                # 这里做一下转换，一天有86400秒
                relative_timestamps = convert_abs_time_to_relative(time_gap_block)
                sample_cn_char_timestamp_mapping = collections.defaultdict(lambda: [])
                for char_i, (cn_char, timestamp) in enumerate(zip(pure_text_block, relative_timestamps)):
                    sample_cn_char_timestamp_mapping[cn_char].append((timestamp, char_i))

            else:
                sample_cn_char_timestamp_mapping = None
                pure_text_block = [x for i, x in enumerate(text_block.split(' ')) if (i + 1) % 3 != 0]
                text_block = ''.join(pure_text_block)

            # Get tokenized ids
            output = tokenizer.encode(text_block)
            offsets = output.offsets
            tokenized_ids = output.ids

            if self.save_temp_samples:
                self.temp_samples.append(tuple(tokenized_ids))

            tokenized_texts = output.tokens
            assert max(tokenized_ids) <= tokenizer.get_vocab_size(), ipdb.set_trace()
            tokenized_ids = tokenized_ids[:block_size]
            if tokenized_texts:
                tokenized_texts = tokenized_texts[:block_size]

            # # Check offsets match with tokenized texts
            # is_valid = True
            # for offset_i, offset in enumerate(offsets):
            #     offset_text = text_block[offset[0]: offset[1]]
            #     tokenized_text = tokenized_texts[offset_i]
            #     tokenized_text = tokenized_text.replace('_', '').replace('▁', '')
            #     if offset_text != tokenized_text:
            #         is_valid = False
            #         break

            if use_time_embed:
                text_hash = md5_time_embedidng(tokenized_texts, {'is_timestamp_rnn': is_timestamp_rnn})
                if text_hash in self.time_embedding_cache:
                    subword_timestamps = self.time_embedding_cache[text_hash]
                    assert (subword_timestamps >= 0).all()
                    assert (subword_timestamps <= 86400).all()
                    subword_tokenized_ids = []
                    for sub_word_i, subword in enumerate(tokenized_texts):
                        subword = subword.replace('_', '').replace('▁', '')
                        if not subword:
                            continue
                        else:
                            subword_tokenized_ids.append(tokenized_ids[sub_word_i])
                else:
                    subword_timestamps = []
                    subword_tokenized_ids = []
                    target_index = 0
                    previous_timestamp = 0
                    for sub_word_i, subword in enumerate(tokenized_texts):
                        subword = subword.replace('_', '').replace('▁', '')
                        if not subword:
                            continue
                        else:
                            subword_tokenized_ids.append(tokenized_ids[sub_word_i])
                        # print("-")
                        # print(f"subword: {subword}")
                        # print("-")
                        subword_timestamp = []
                        for word in subword:
                            valid_timestamps = sample_cn_char_timestamp_mapping[word]
                            valid_timestamp_indices = np.array([x[1] for x in valid_timestamps])
                            closest_index = np.argmin(np.abs(valid_timestamp_indices - target_index))
                            closest_gap = np.min(np.abs(valid_timestamp_indices - target_index))
                            timestamp = valid_timestamps[closest_index][0]
                            if timestamp < previous_timestamp:
                                timestamp = previous_timestamp
                            # print(
                            #     f"word: {word}, timestamp: {timestamp}, closest_gap: {closest_gap}")
                            subword_timestamp.append(timestamp)
                            previous_timestamp = timestamp
                            target_index += 1
                        if is_timestamp_rnn:
                            subword_timestamp = tuple(subword_timestamp)
                            subword_timestamps.append(tuple(subword_timestamp))
                        else:
                            subword_timestamp = int(np.average(subword_timestamp))
                            subword_timestamps.append(subword_timestamp)

                    # Upadate for time embedding cache
                    subword_timestamps_arr = np.array(subword_timestamps).astype(np.int32)
                    assert (subword_timestamps_arr >= 0).all()
                    assert (subword_timestamps_arr <= 86400).all()
                    self.time_embedding_cache[text_hash] = subword_timestamps_arr

                subword_tokenized_ids = np.array(subword_tokenized_ids)
                subword_timestamps = tuple(subword_timestamps)
                assert subword_tokenized_ids.shape[0] == len(subword_timestamps)
                example = subword_tokenized_ids
            else:
                subword_timestamps = None
                example = np.array(tokenized_ids)
            # --------------------------------------------------------------------------------------------------

            if len(example) < block_size:
                # pad example
                all_pad_example = np.full(block_size, tokenizer.pad_token_id)
                all_pad_example[:len(example)] = example
                example = all_pad_example

                # if use_time_embed:
                #
                #     if use_bpe and use_sinusoidal:
                #         pass
                #     else:
                #         all_pad_time_gap = np.full(block_size, 0)
                #         all_pad_time_gap[:len(time_gaps)] = time_gaps
                #         time_gaps = all_pad_time_gap

            # add example
            self.examples.append(example)
            if use_time_embed:
                self.time_gaps.append(subword_timestamps)

        if self.use_time_embed:
            # save embedding cache
            pickle.dump(self.time_embedding_cache, open(self.cache_save_path, 'wb'))
            print(f"[Time embedding cache] save to {self.cache_save_path} SUCCESS")

        # Save temp samples to local
        if self.save_temp_samples:
            temp_sample_save_path = '../static/temp_samples.txt'
            with open(temp_sample_save_path, 'w') as f:
                for sample in self.temp_samples:
                    f.write('\t'.join([str(x) for x in sample]) + '\n')

            print(f"Save temp samples to {temp_sample_save_path}, total: {len(self.temp_samples)}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Mode: General
        if self.use_time_embed:
            # torch.Size([512]),
            return torch.tensor(self.examples[i], dtype=torch.long), self.time_gaps[i]
        else:
            # return torch.tensor(self.examples[i], dtype=torch.long)
            return torch.tensor(self.examples[i], dtype=torch.long), None


# -------------

def get_dataset_bpe(
        args: DataTrainingArguments,
        tokenizer,
        use_time_embed=False,
        debugN=None,
        hdf_data=None,
        is_timestamp_rnn=False,
        embedding_cache_dir=None,
):
    hdf5_file_path = args.h5_data_file_path

    return BpeTextDataset(
        hdf_data=hdf_data,
        tokenizer=tokenizer,
        hdf5_file_path=hdf5_file_path,
        block_size=args.block_size,
        use_time_embed=use_time_embed,
        debugN=debugN,
        is_timestamp_rnn=is_timestamp_rnn,
        cache_dir=embedding_cache_dir
    )
