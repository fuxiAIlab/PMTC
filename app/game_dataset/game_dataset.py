"""
This is a compact training script for game bert
"""

import logging
import math
import numpy as np
import os
import ipdb
import torch
import time
import sys

sys.path.append('..')
import datetime

from torch.utils.data.dataset import Dataset

from typing import Optional
from tqdm import tqdm
from args_utils import ModelArguments, DataTrainingArguments

# Init logger
logger = logging.getLogger(__name__)

from transformers import (
    PreTrainedTokenizer,
)




class TextDataset(Dataset):
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
            overwrite_cache=False,
            cache_dir: Optional[str] = None,
            use_time_embed: bool = False,
            use_bpe=False,
            debugN=None,
            max_time_gap=None,
            use_sinusoidal=False,
            behave_tokenizer=None,
            design_tokenizer=None,
    ):

        if tokenizer is None:
            assert use_time_embed
            assert behave_tokenizer and design_tokenizer
            assert not use_bpe
        self.tokenizer = tokenizer
        self.behave_tokenizer = behave_tokenizer
        self.design_tokenizer = design_tokenizer

        self.use_time_embed = use_time_embed
        assert os.path.isfile(hdf5_file_path), f"Input file path {hdf5_file_path} not found"

        self.examples = []
        self.time_gaps = []
        self.design_ids = []

        for sample_i, sample in enumerate(hdf_data):

            if debugN:
                if sample_i >= debugN:
                    print(f"[DEBUG N] Stop loading data, debug N is set to {debugN}")
                    break
            sample = [x for x in sample if x != '[PAD]']
            text_block = ' '.join(sample)
            # --------------------------------------------------------------------------------------------------
            # CODE BY PJS
            # --------------------------------------------------------------------------------------------------

            if use_time_embed:
                pure_text_block = [x for i, x in enumerate(text_block.split(' ')) if (i + 1) % 3 != 0]
                time_gap_block = text_block.split(' ')[2::3]

                # This is only for data assertion
                temp_test_time_gap = time_gap_block[0]
                temp_date_obj = datetime.datetime.fromtimestamp(int(temp_test_time_gap))

                assert 2019 < temp_date_obj.year < 2022

                # compute the time gap if sinusoidal is not used
                if not use_sinusoidal:
                    # TODO, 这个地方我觉得还是要改一下，统一成最大值1024秒，最小单位是秒，但是最小的单位是1秒*100
                    time_gap_block = list(zip(time_gap_block, time_gap_block[1:] + [0]))
                    time_gap_block = [math.ceil((int(t2) - int(t1)) / 100) for t1, t2 in time_gap_block]
                    time_gap_block[-1] = 0
                    assert min(time_gap_block) >= 0

                if tokenizer is not None:
                    time_gap_block = [y for x in zip(time_gap_block, time_gap_block) for y in x][:block_size]
                else:
                    time_gap_block = [y for x in zip(time_gap_block, time_gap_block) for y in x][:int(block_size * 2)]

                if use_bpe:
                    text_block = ''.join(pure_text_block)
                else:
                    text_block = ' '.join(pure_text_block)

            else:
                pure_text_block = [x for i, x in enumerate(text_block.split(' ')) if (i + 1) % 3 != 0]
                if use_bpe:
                    text_block = ''.join(pure_text_block)
                else:
                    text_block = ' '.join(pure_text_block)

            if tokenizer is not None:
                output = tokenizer.encode(text_block)
                tokenized_ids = output.ids
                tokenized_texts = output.tokens
                if not use_bpe:
                    assert max(tokenized_ids) <= tokenizer.max_vocab_index, ipdb.set_trace()
                else:
                    assert max(tokenized_ids) <= tokenizer.get_vocab_size(), ipdb.set_trace()
                design_tokenized_ids = None
            else:

                # get behave token
                behave_output = behave_tokenizer.encode(' '.join(text_block.split()[::2]))
                behave_tokenized_ids = behave_output.ids
                behave_texts = behave_output.tokens

                # get design token
                design_output = design_tokenizer.encode(' '.join(text_block.split()[1::2]))
                design_tokenized_ids = design_output.ids
                design_texts = design_output.tokens

                # combine them all
                assert len(behave_tokenized_ids) == len(design_tokenized_ids) == len(behave_texts) == len(design_texts)
                tokenized_ids = behave_tokenized_ids
                tokenized_texts = None

            tokenized_ids = tokenized_ids[:block_size]
            if design_tokenized_ids:
                design_tokenized_ids = design_tokenized_ids[:block_size]

            if tokenized_texts:
                tokenized_texts = tokenized_texts[:block_size]

            example = np.array(tokenized_ids)

            if use_time_embed:
                time_gaps = np.array([int(x) for x in time_gap_block], dtype=int)
                if use_bpe:
                    if use_sinusoidal:

                        # 这里做一下转换，一天有86400秒
                        time_gap0 = datetime.datetime.fromtimestamp(time_gaps[0])
                        today_start = datetime.datetime(year=time_gap0.year, month=time_gap0.month, day=time_gap0.day)
                        today_start_timestamp = int(time.mktime(today_start.timetuple()))
                        time_gaps = time_gaps - today_start_timestamp

                        new_time_gaps = []
                        start_index = 0
                        for word in tokenized_texts:
                            word = word.replace('_', '').replace('▁', '')
                            new_time = time_gaps[start_index:start_index + len(word)]
                            new_time_gaps.append(tuple(new_time))
                            start_index += len(word)
                        time_gaps = np.array(new_time_gaps, dtype=object)
                    else:
                        new_time_gaps = []
                        start_index = 0
                        for word in tokenized_texts:
                            word = word.replace('_', '').replace('▁', '')
                            new_time_gap = time_gaps[start_index:start_index + len(word)]
                            new_time_gaps.append(sum(new_time_gap))
                            start_index += len(word)
                        new_time_gaps = np.array(new_time_gaps)

                # cut off max time gap
                if not use_sinusoidal:
                    time_gaps = np.array([x if x <= max_time_gap - 1 else max_time_gap - 1 for x in time_gaps])

                # recover the length of time gaps for sperate ids
                if tokenizer is None:
                    time_gaps = time_gaps[::2]

                assert example.shape == time_gaps.shape

            if tokenizer is None:
                assert example.shape == time_gaps.shape == np.array(design_tokenized_ids).shape
            # --------------------------------------------------------------------------------------------------

            if len(example) < block_size:

                # pad example
                if tokenizer:
                    all_pad_example = np.full(block_size, tokenizer.pad_token_id)
                else:
                    all_pad_example = np.full(block_size, behave_tokenizer.pad_token_id)
                all_pad_example[:len(example)] = example
                example = all_pad_example

                # pad design_id
                if design_tokenized_ids:
                    all_pad_design_ids = np.full(block_size, design_tokenizer.pad_token_id)
                    all_pad_design_ids[:len(design_tokenized_ids)] = design_tokenized_ids
                    design_tokenized_ids = all_pad_design_ids

                if use_time_embed:
                    if use_bpe and use_sinusoidal:
                        pass
                    else:
                        all_pad_time_gap = np.full(block_size, 0)
                        all_pad_time_gap[:len(time_gaps)] = time_gaps
                        time_gaps = all_pad_time_gap

            # add example
            self.examples.append(example)

            # add design id
            if not tokenizer:
                self.design_ids.append(np.array(design_tokenized_ids))

            if use_time_embed:
                self.time_gaps.append(time_gaps)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):

        # Mode: three in one
        if self.tokenizer is None:
            cat_arr = np.concatenate([self.examples[i], self.design_ids[i], self.time_gaps[i]])
            return torch.tensor(cat_arr, dtype=torch.long)
        else:
            # Mode: General
            if self.use_time_embed:
                cat_arr = np.concatenate([self.examples[i], self.time_gaps[i]])
                return torch.tensor(cat_arr, dtype=torch.long)
            else:
                return torch.tensor(self.examples[i], dtype=torch.long)


# -------------

def get_dataset(
        args: DataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        behave_tokenizer=None,
        design_tokenizer=None,
        evaluate: bool = False,
        cache_dir: Optional[str] = None,
        use_time_embed=False,
        debugN=None,
        hdf_data=None,
        use_bpe=False,
        max_time_gap=None,
        use_sinusoidal=False
):
    hdf5_file_path = args.h5_data_file_path

    return TextDataset(
        hdf_data=hdf_data,
        tokenizer=tokenizer,
        behave_tokenizer=behave_tokenizer,
        design_tokenizer=design_tokenizer,
        hdf5_file_path=hdf5_file_path,
        block_size=args.block_size,
        overwrite_cache=args.overwrite_cache,
        cache_dir=cache_dir,
        use_time_embed=use_time_embed,
        debugN=debugN,
        use_bpe=use_bpe,
        max_time_gap=max_time_gap,
        use_sinusoidal=use_sinusoidal
    )
