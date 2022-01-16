import collections
import ipdb
import copy
import random
import numpy as np
from sklearn.metrics import f1_score as compute_f1_score
from sklearn.metrics import ndcg_score as compute_ndcg_score


class TrainMetric:
    def __init__(self):
        pass


class MacroF1Metric(TrainMetric):
    def __init__(self):
        super().__init__()
        self._reset_y()
        self.name = 'macro_f1'

    def _reset_y(self):
        self.predict_labels = []
        self.true_labels = []

    def extend_ys(self, predict_ys, true_ys):
        assert len(predict_ys) == len(true_ys)
        self.predict_labels.extend(predict_ys)
        self.true_labels.extend(true_ys)

    def append_y(self, predict_y, true_y):
        self.predict_labels.append(predict_y)
        self.true_labels.append(true_y)

    def compute_metric_etc(self):
        macro_f1 = compute_f1_score(self.true_labels, self.predict_labels, average='macro')
        # compute shuffle f1
        shuffle_y = copy.deepcopy(self.true_labels)
        random.shuffle(shuffle_y)
        shuffle_macro_f1 = compute_f1_score(self.true_labels, shuffle_y, average='macro')
        self._reset_y()
        return macro_f1, shuffle_macro_f1

    def compute_metric(self):
        macro_f1, _ = self.compute_metric_etc()
        return macro_f1


class NDCGMetric(TrainMetric):
    def __init__(self):
        super().__init__()
        self._reset_y()
        self.name = 'NDCG'

    def _reset_y(self):
        ndcg_prob_dict = {'output_prob': [], 'true_prob': [], 'random_prob': []}
        self.ndcg_dict = collections.defaultdict(lambda: copy.deepcopy(ndcg_prob_dict))

    def update_y(self, role_id_dses, true_probs, output_probs, random_probs):
        for role_id_ds, true_prob, output_prob, random_prob in zip(role_id_dses,
                                                                   true_probs,
                                                                   output_probs,
                                                                   random_probs):
            self.ndcg_dict[role_id_ds]['true_prob'].append(float(true_prob))
            self.ndcg_dict[role_id_ds]['output_prob'].append(float(output_prob))
            self.ndcg_dict[role_id_ds]['random_prob'].append(float(random_prob))

    def compute_metric_etc(self):
        ndcg_scores = []
        ndcg_scores_random = []
        for role_id_ds, prob_dict in self.ndcg_dict.items():
            if len(prob_dict['true_prob']) == 1:
                continue
            else:
                one_ndcg_score = compute_ndcg_score([prob_dict['true_prob']], [prob_dict['output_prob']])
                one_ndcg_score_random = compute_ndcg_score([prob_dict['true_prob']], [prob_dict['random_prob']])
                ndcg_scores.append(one_ndcg_score)
                ndcg_scores_random.append(one_ndcg_score_random)

        if ndcg_scores:
            ndcg_score = np.average(ndcg_scores)
            ndcg_score_random = np.average(ndcg_scores_random)
        else:
            ndcg_score, ndcg_score_random = 0.0, 0.0
        self._reset_y()
        return ndcg_score, ndcg_score_random, ndcg_scores

    def compute_metric(self):
        ndcg_score, _, _ = self.compute_metric_etc()
        return ndcg_score
