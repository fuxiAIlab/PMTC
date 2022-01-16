import random
import textdistance
import collections
import torch

import copy
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from sklearn.metrics import adjusted_rand_score


def _partition(list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]


def _compute_edit_distance(group: List[List[str]]):
    eds = []
    for i1, sample1 in enumerate(group):
        for i2, sample2 in enumerate(group):
            if i1 == i2:
                continue
            else:
                ed = textdistance.levenshtein.distance(sample1, sample2)
                eds.append(ed)
    min_ed, max_ed, avg_ed = np.min(eds), np.max(eds), np.average(eds)
    return min_ed, max_ed, avg_ed


def _compute_jaccard_distance(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


class ListDistance:
    def __init__(self, max_len=256):
        self.max_len = max_len

    def compute_distances_for_groups(self,
                                     groups: List[List[List[str]]]):
        groups_distances = collections.defaultdict(lambda: {'min': [], 'max': [], 'avg': []})
        for group in tqdm(groups, total=len(groups)):
            if len(group) == 1:
                continue
            else:
                group_distances = self.compute_distances_for_group(group)
                for distance_key, (min_v, max_v, avg_v) in group_distances.items():
                    groups_distances[distance_key]['min'].append(min_v)
                    groups_distances[distance_key]['max'].append(max_v)
                    groups_distances[distance_key]['avg'].append(avg_v)
        return groups_distances

    def compute_distances_for_group(self,
                                    group: List[List[str]]):
        distances = {'ed': [], 'jaccard': []}
        for i1, sample1 in tqdm(enumerate(group), total=len(group)):
            for i2, sample2 in enumerate(group):
                if i2 <= i1:
                    continue
                else:
                    ed = textdistance.levenshtein.distance(sample1[:self.max_len], sample2[:self.max_len])
                    jaccard = _compute_jaccard_distance(sample1[:self.max_len], sample2[:self.max_len])
                    distances['ed'].append(ed)
                    distances['jaccard'].append(jaccard)

        for distance_type, values in distances.items():
            distances[distance_type] = (np.min(values), np.max(values), np.average(values))

        return distances


class EmbedderChecker:

    def __init__(self,
                 seed: int = 1,
                 max_sample_size: int = 100
                 ):
        # set random seed
        random.seed(seed)
        np.random.seed(seed)

        self.max_sample_size = max_sample_size
        self.ed_checker = ListDistance()

    def _build_vocab(self,
                     input):
        vocab = set()
        for x in input:
            vocab.update(set(x))
        vocab = sorted(list(vocab))
        return vocab

    def consistent_check(self,
                         embed_func: Callable,
                         input: List[List[str]],
                         n: int = 10):
        repeat = 3
        input_sample = random.sample(input, min(n, len(input)))

        previous_embedding = None
        check_valid = True
        for i in range(repeat):
            embedding = embed_func(input_sample)
            if previous_embedding is not None:
                check_valid = bool((embedding == previous_embedding).all())
                if not check_valid:
                    break
            previous_embedding = embedding

        return check_valid

    def random_batch_consistent_check(self,
                                      embed_func: Callable,
                                      input: List[List[str]],
                                      sample_n: int = 10,
                                      duplicate_n: int = 2,
                                      conca_output_tasks: Union[List, Tuple] = None
                                      ):
        input_sample = random.sample(input, min(sample_n, len(input)))
        input_sample_indices = list(range(len(input_sample)))

        # randomly add samples to new list, thus create duplicated samples, to test whether embedding remains the same
        # when sample is in different batch

        new_input_sample = []
        origin_indices = collections.defaultdict(lambda: [])

        # Duplicate samples by a factor of 10
        for i in range(duplicate_n * sample_n):
            random_index = random.choice(input_sample_indices)
            random_sample = input_sample[random_index]
            new_input_sample.append(random_sample)
            origin_indices[random_index].append(i)

        embeddings = embed_func(new_input_sample, conca_output_tasks=conca_output_tasks)

        # check sample input always has same output
        for origin_index, indices in origin_indices.items():
            indices_embedding = embeddings[indices]
            unique_n = torch.unique(indices_embedding, dim=0).shape[0]
            if unique_n > 1:
                print(f"group-{origin_index}, find different embeddings for the same input, unique_n: {unique_n}!")
                return False

        return True

    def _default_perturb_func(self,
                              sample: List[str],
                              vocab: List,
                              perturb_n: int):
        valid_indices = list(range(len(sample)))
        perturbed_samples = []

        for i in range(perturb_n):
            sample_copy = copy.deepcopy(sample)
            sample_copy[random.choice(valid_indices)] = random.choice(vocab)
            perturbed_samples.append(sample_copy)

        return perturbed_samples

    def do_kmeans_from_embedding(self,
                                 input_embedding: Union[np.ndarray, torch.Tensor],
                                 input_sample: List[List[str]],
                                 n_clusters: int):
        if isinstance(input_embedding, torch.Tensor):
            input_embedding = input_embedding.cpu().numpy()

        # Get K-means group
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, verbose=2, max_iter=2000, tol=1e-6).fit(input_embedding)
        kmeans_labels = kmeans.labels_
        kmeans_groups = collections.defaultdict(lambda: [])
        for i, label in enumerate(kmeans_labels):
            kmeans_groups[label].append(input_sample[i])
        kmeans_groups = list(kmeans_groups.values())
        return kmeans_labels, kmeans_groups

    def clustering_check(self,
                         embed_func,
                         input: List[List[str]],
                         vocab_func: Callable = None,
                         perturb_func: Callable = None,
                         perturb_n: int = 5,
                         conca_output_tasks: Union[List, Tuple] = None
                         ) -> float:
        """
        Check by adding small perturbations to input, and then do clustering
        """

        input_sample = random.sample(input, min(self.max_sample_size, len(input)))

        to_cluster_samples = []
        to_cluster_labels = []
        to_cluster_label_i = 0

        # create vocab
        if vocab_func is None:
            vocab = self._build_vocab(input)
        else:
            vocab = vocab_func(input)

        # set default perturb func
        if perturb_func is None:
            perturb_func = self._default_perturb_func

        # Get perturb samples
        for sample in input_sample:
            perturbed_samples = perturb_func(sample, vocab, perturb_n)

            # add origin
            to_cluster_samples.append(sample)
            to_cluster_labels.append(to_cluster_label_i)

            # add perturb
            to_cluster_samples.extend(perturbed_samples)
            to_cluster_labels.extend([to_cluster_label_i for _ in perturbed_samples])
            to_cluster_label_i += 1

        # do clustering
        to_cluster_embedding = embed_func(to_cluster_samples, conca_output_tasks=conca_output_tasks)
        kmeans_labels, kmeans_groups = self.do_kmeans_from_embedding(to_cluster_embedding,
                                                                     to_cluster_samples,
                                                                     len(input_sample))

        # compute ari score for clustering
        ari_score = adjusted_rand_score(to_cluster_labels, kmeans_labels)
        return ari_score

    def distance_check(self,
                       embed_func: Callable,
                       input: List[List[str]],
                       n_clusters: int = 20
                       ):
        """
        Include edit distance (The lower, the better) & jaccard_distance (The higher, the better)
        """

        input_sample = random.sample(input, min(self.max_sample_size, len(input)))
        print(f"Resample Input done, Size: {len(input_sample)}")

        input_embedding = embed_func(input_sample)

        # do kmeans
        kmeans_labels, kmeans_groups = self.do_kmeans_from_embedding(input_embedding, input_sample, n_clusters)

        #                     distances['ed'].append(ed)
        #                     distances['jaccard'].append(jaccard)

        kmeans_eds = self.ed_checker.compute_distances_for_groups(kmeans_groups)
        kmeans_avg_ed = np.average(kmeans_eds['ed']['avg'])
        kmeans_avg_jaccard = np.average(kmeans_eds['jaccard']['avg'])

        # Get random group, split input_sample randomly into n_clusters parts
        random_groups = _partition(input_sample, n_clusters)
        random_eds = self.ed_checker.compute_distances_for_groups(random_groups)
        random_avg_ed = np.average(random_eds['ed']['avg'])
        random_avg_jaccard = np.average(random_eds['jaccard']['avg'])

        result = {
            'Kmeans_group_edit_distance': kmeans_avg_ed,
            'random_group_edit_distance': random_avg_ed,
            'Kmeans_group_jaccard_distance': kmeans_avg_jaccard,
            'random_group_jaccard_distance': random_avg_jaccard,
            'is_Kmeans_better': True if (kmeans_avg_ed < random_avg_ed) and (
                    kmeans_avg_jaccard > random_avg_jaccard) else False
        }
        return result

    def check_all(self,
                  embed_func: Callable,
                  input: List[List[str]]):

        # check consistency
        consistency_pass = self.consistent_check(embed_func, input)
        if consistency_pass:
            print("Consitency Check PASS ✓")
        else:
            print("Consitency Check FAILED X")
            return
