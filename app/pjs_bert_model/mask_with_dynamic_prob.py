import os
import ipdb
import json
import torch
import numpy as np
import collections


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


# dynamic_mask_predict_prob

class DynamicMaskerRecorder:
    def __init__(self,
                 mask_type,
                 idf_path=None,
                 record_snapshot=False,
                 is_record_mask_ratio=False):
        if mask_type in {'posterior_prob', 'lowest_prob', 'part_prob', 'part_prob_linear_increase'}:
            self.prob_tensor = torch.zeros((50000,))
            self.idf_tensor = None
            self.tf_idf_warm_up_probs = None
        elif mask_type == 'tf_idf':
            self.idf_tensor = torch.zeros((50000,))
            self.prob_tensor = torch.zeros((50000,))
            idf_dict = load_save_json(idf_path, 'load')
            for index, idf in idf_dict.items():
                self.idf_tensor[int(index)] = idf
            self.tf_idf_warm_up_probs = None
        elif mask_type == 'posterior_prob_with_tf_idf_warmup':
            self.prob_tensor = torch.zeros((50000,))
            self.idf_tensor = torch.zeros((50000,))
            idf_dict = load_save_json(idf_path, 'load')
            for index, idf in idf_dict.items():
                self.idf_tensor[int(index)] = idf
            self.tf_idf_warm_up_probs = []
        else:
            self.tf_idf_warm_up_probs = None
            self.idf_tensor = None
            if record_snapshot:
                self.prob_tensor = torch.zeros((50000,))

        self.record_snapshot = record_snapshot
        if record_snapshot:
            self.step_mask_probabilities = []
            self.step_sample_mask_distributions = {'train_step': [],
                                                   'tokens': [],
                                                   'mask_prob': [],
                                                   'softmax_t': [],
                                                   'avg_model_prob': []}
            self._prob_snapshots = []
            self._mask_distribution_snapshots = []
        else:
            self._prob_snapshots = None
            self._mask_distribution_snapshots = None

        if self.idf_tensor is not None:
            # <s>,0,0,0.0,tf_idf,0.0001,0.5,350709,0
            # <pad>,1,0,0.0,tf_idf,0.0001,0.5,350709,0
            # </s>,2,0,0.0,tf_idf,0.0001,0.5,350709,0
            # <unk>,3,0,0.0,tf_idf,0.0001,0.5,350709,0
            # <mask>,4,0,0.0,tf_idf,0.0001,0.5,350709,0
            # ▁,5,0,0.0,tf_idf,0.0001,0.5,350709,0
            # 这里要保证某些没有出现过的token也有idf值，就直接设成最大的
            self.idf_tensor[self.idf_tensor == 0.0] = max(self.idf_tensor)
            self.idf_tensor[:5] = -1  # 前10个好像是碰不到的token,就先设成没有物理意义的-1
            self.idf_tensor[5:10] = 3  # TODO, 之前计算的idf vocab有点问题，目前就用2左右代替
            self.idf_tensor[3] = int(torch.min(self.idf_tensor[5:]))  # 这个是unk的token

        self.token_last_predict_prob = torch.zeros((50000,))
        self.token_freq = torch.zeros((50000,), dtype=torch.int)
        self.total_mask_label_count = 0
        self.train_step = 0
        self.overshoot_count = 0
        self.label_maskes = []
        self.total_steps = None

        if is_record_mask_ratio:
            self.mask_ratios = []
        else:
            self.mask_ratios = None

    @property
    def mean_prob_tensor(self):
        return float(torch.mean(self.prob_tensor))

    @property
    def prob_snapshots(self):
        return self._prob_snapshots

    @property
    def mask_distribution_snapshots(self):
        return self._mask_distribution_snapshots

    def reset_step_mask_probabilities(self):
        self.step_mask_probabilities = []

    def add_snapshot(self, step, t):
        self._prob_snapshots.append((step, t.clone()))
        histogram_count, histogram_ranges = np.histogram(self.step_mask_probabilities, bins=10, range=(0, 1))
        total = np.sum(histogram_count)
        histogram_count = histogram_count / total
        histogram_count = tuple(histogram_count)
        self._mask_distribution_snapshots.append((step, histogram_count))
