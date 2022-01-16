import ipdb
import torch
import collections
import random
from torch import nn
from torch.utils.data import Dataset
from pytorch_lightning import Trainer, seed_everything


class ChurnPredictToyDataset(Dataset):
    def __init__(self, data_size, seq_len, vocab_size, input_size,
                 mask_zero_prob=0.05,
                 use_transformer_feature=True,
                 transformer_dim=768):
        random_embedding = nn.Embedding(vocab_size + 1, input_size)
        self.examples = []
        for i in range(data_size):
            x = torch.tensor([random.randint(1, vocab_size) for _ in range(seq_len)])
            x_indices = torch.tensor(range(len(x)))
            zero_prob = torch.full_like(x_indices, mask_zero_prob, dtype=torch.float)
            zero_mask = torch.bernoulli(zero_prob).bool()
            mask_indices = x_indices[zero_mask]
            x[mask_indices] = 0
            zero_count = int(torch.bincount(x)[0])
            if zero_count >= 1:
                y = 1
            else:
                y = 0
            embedding_x = random_embedding(x).detach()
            if use_transformer_feature:
                transformer_embedding = torch.zeros((seq_len, transformer_dim))
            else:
                transformer_embedding = torch.empty((seq_len, transformer_dim))

            self.examples.append((embedding_x, transformer_embedding, torch.tensor(y)))
        print(collections.Counter([int(x[2]) for x in self.examples]))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class ToyDataset1(Dataset):
    def __init__(self, data_size, vocab_size, max_seq_len, padding_index, transformer_input_size,
                 use_transformer_feature):
        seed_everything(1)
        self.examples = []

        int_pool = list(range(max_seq_len))
        if padding_index in int_pool:
            int_pool.remove(0)

        for i in range(data_size):
            x_pad = torch.full((max_seq_len,), padding_index, dtype=torch.long)
            x_len = random.choice(int_pool)
            x = torch.randint(0, vocab_size, (x_len,))
            x_pad[:len(x)] = x

            # # 用和的奇偶性来决定y
            # if int(sum(x)) % 2 == 0:
            #     y = torch.tensor(0)
            # else:
            #     y = torch.tensor(1)

            # # 用最后一位的奇偶性来决定y
            # if int(x[-1]) % 2 == 0:
            #     y = torch.tensor(0)
            # else:
            #     y = torch.tensor(1)

            # # 用第一位的奇偶性来决定y
            # if int(x[0]) % 2 == 0:
            #     y = torch.tensor(0)
            # else:
            #     y = torch.tensor(1)

            # # 用最后一位和第一位的和的奇偶性来决定y
            # if int(x[-1] + x[0]) % 2 == 0:
            #     y = torch.tensor(0)
            # else:
            #     y = torch.tensor(1)

            # 用最后一位和第一位的和的奇偶性来决定y
            if int(x[-1]) % 2 == 0 or int(x[0]) % 2 == 0:
                y = torch.tensor(0)
            else:
                y = torch.tensor(1)

            # add transformer features
            if use_transformer_feature:
                transformer_x = torch.zeros((transformer_input_size,))
            else:
                transformer_x = torch.empty((transformer_input_size,))

            self.examples.append(((x_pad, torch.tensor(x_len)), transformer_x, y))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class MapPreloadDataset1(Dataset):
    def __init__(self, data_size, vocab_size, max_seq_len, padding_index, transformer_input_size,
                 use_transformer_feature):
        seed_everything(1)
        self.examples = []

        int_pool = list(range(max_seq_len))
        if padding_index in int_pool:
            int_pool.remove(0)

        for i in range(data_size):

            x_pad = torch.randint(0, vocab_size, (max_seq_len,))

            # 用前面出现次数最多的输入作为输出
            x_counter = list(collections.Counter(x_pad.tolist()).items())
            x_counter = sorted(x_counter, key=lambda x: x[1], reverse=True)
            y = torch.tensor([x_counter[0][0]])

            # add transformer features
            if use_transformer_feature:
                transformer_x = torch.zeros((transformer_input_size,))
            else:
                transformer_x = torch.empty((transformer_input_size,))

            self.examples.append(((x_pad, torch.tensor(max_seq_len)), transformer_x, y))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn_variable_len(batch):
    batch_x_pad = []
    batch_x_len = []
    batch_transformer_embedding = []
    batch_y = []
    for ((x_pad, x_len), transformer_x, y) in batch:
        batch_x_pad.append(x_pad)
        batch_x_len.append(x_len)
        batch_transformer_embedding.append(transformer_x)
        batch_y.append(y)

    # Sort from longest to shortest
    batch_x_len = torch.stack(batch_x_len)
    sort_indices = torch.argsort(batch_x_len, descending=True)

    batch_x_pad = torch.stack(batch_x_pad)[sort_indices]
    batch_x_len = batch_x_len[sort_indices]
    batch_transformer_embedding = torch.stack(batch_transformer_embedding)[sort_indices]
    batch_y = torch.stack(batch_y)[sort_indices]

    return ((batch_x_pad, batch_x_len), batch_transformer_embedding, batch_y)
