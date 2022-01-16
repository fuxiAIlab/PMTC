import ipdb
import torch
from torch import nn


class BpeTimeStampRnn(nn.Module):
    def __init__(self,
                 input_size=768,
                 hidden_size=768,
                 num_layers=1,
                 ):
        super().__init__()
        self.hidden_size = int(hidden_size / 2)
        self.rnn = nn.LSTM(input_size, self.hidden_size, num_layers, batch_first=True, bidirectional=True)

    def forward(self, embedding_x, lens_x):
        lens_x = torch.tensor(lens_x)

        output, _ = self.rnn(embedding_x)

        # Forward, Here We use fancy indexing
        forward_outputs = output[list(range(output.shape[0])), lens_x - 1][:, :self.hidden_size]
        # Backward
        backward_outputs = output[list(range(output.shape[0])), [0 for _ in lens_x]][:, self.hidden_size:]
        # Merge
        outputs = torch.cat([forward_outputs, backward_outputs], dim=1)

        return outputs
