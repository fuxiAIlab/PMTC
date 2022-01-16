import os
import json
import random
import numpy as np
import torch


def set_randseed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_save_json(json_path, mode, verbose=1, encoding='utf-8', data=None):
    if mode == 'save':
        assert data is not None
        with open(json_path, 'w', encoding=encoding) as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            if verbose >= 1:
                print(f"save json data to {json_path}")
    elif mode == 'load':
        if os.path.isfile(json_path):
            with open(json_path, 'r', encoding=encoding) as f:
                response = json.load(f)
            if verbose >= 1:
                print(f"load json from {json_path} success")
        else:
            raise Exception(f"{json_path} does not exist!")
        return response
    else:
        raise NotImplementedError


def mask_prob_masking_post_process(masked_indices, attention_masks, probability_matrix):
    valid_masked_indices = []
    for mask_index_i, mask_index in enumerate(masked_indices):
        while True:
            if (mask_index[attention_masks[mask_index_i].bool()] == False).any():
                break
            else:
                mask_index = torch.bernoulli(probability_matrix[mask_index_i]).bool()
        valid_masked_indices.append(mask_index)
    valid_masked_indices = torch.stack(valid_masked_indices)
    return valid_masked_indices
