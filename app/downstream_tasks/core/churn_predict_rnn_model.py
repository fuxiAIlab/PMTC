import ipdb
import copy
import random
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from core.train_metric import MacroF1Metric
from .config import train_config

MLP_HIDDEN_SIZE1 = 256
MLP_HIDDEN_SIZE2 = 256
RNN_HIDDEN_SIZE = 256


class ChurnPredictRnnModel(pl.LightningModule):
    def __init__(self,
                 handfeature_dim,
                 mlp_hidden_size1=MLP_HIDDEN_SIZE1,
                 mlp_hidden_size2=MLP_HIDDEN_SIZE2,
                 rnn_hidden_size=RNN_HIDDEN_SIZE,
                 transformer_input_size=768,
                 use_pretrain_features=False,
                 use_base_features=True,
                 rnn_num_layer=1,
                 lr=train_config.lr,
                 transformer_encoder_params=None
                 ):
        super().__init__()
        self.use_pretrain_features = use_pretrain_features
        self.use_base_features = use_base_features

        # Transformer Input projection layer
        self.input_projection = nn.Linear(transformer_input_size, 768)

        # Init Rnn
        if use_base_features and use_pretrain_features:
            rnn_input_dim = handfeature_dim + self.input_projection.out_features
        elif use_base_features:
            rnn_input_dim = handfeature_dim
        elif use_pretrain_features:
            rnn_input_dim = self.input_projection.out_features
        else:
            raise Exception
        self.rnn = ChurnPredictRnn(rnn_input_dim, rnn_hidden_size, num_layers=rnn_num_layer)

        # Init Mlp
        mlp_input_size = 2 * rnn_hidden_size  # bi directional
        self.mlp = ChurnPredictMLP(mlp_input_size, mlp_hidden_size1, mlp_hidden_size2)

        self.loss_func = F.binary_cross_entropy
        self.lr = lr
        self.transformer_encoder_params = transformer_encoder_params
        self.is_select_model_by_val_metric = False
        self.val_metric = MacroF1Metric()
        self.test_metric = MacroF1Metric()
        print(self)

    def forward(self, x):
        embedding = self.rnn(x)
        output = self.mlp(embedding)
        return output

    def forward_loss(self, batch, loss_msg):
        x, transformer_embedding, y = batch
        feature_dim = x.shape[2]

        mean_x = torch.mean(x.view(-1, x.shape[2]), dim=0).detach()
        std_x = torch.std(x.view(-1, x.shape[2]), dim=0).detach()
        std_x_to_low_index = torch.where(std_x < 1e-6)[0]
        too_low_mask = torch.zeros((feature_dim,)).bool()
        too_low_mask[std_x_to_low_index] = True
        too_low_mask = too_low_mask.expand(x.shape[0], x.shape[1], too_low_mask.shape[0])
        x = (x - mean_x) / std_x
        x[too_low_mask] = 0.0

        assert not torch.isinf(x).any()
        assert not torch.isnan(x).any()

        # x: torch.Size([8, 14, 194])
        if self.use_pretrain_features:
            assert transformer_embedding.shape[1] != 0
            x = torch.cat([x, self.input_projection(transformer_embedding)], dim=2)

        output = self(x)

        loss = self.loss_func(output, y.float())
        self.log(loss_msg, loss, prog_bar=True)
        return loss, output

    def training_step(self, batch, batch_idx):
        loss, _ = self.forward_loss(batch, 'train_loss')
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self.forward_loss(batch, 'val_loss')
        return loss

    def test_step(self, batch, batch_idx, label1_thresh=0.5):
        loss, output = self.forward_loss(batch, 'test_loss')
        _, _, y = batch

        # compute the f1-score
        predict_labels = torch.zeros_like(output, dtype=torch.long)
        predict_labels[output > label1_thresh] = 1

        # compute labels
        true_labels = y.flatten().detach().cpu().tolist()
        predict_labels = predict_labels.flatten().detach().cpu().tolist()
        self.log('test_loss', loss, prog_bar=True)
        return loss, (true_labels, predict_labels)

    def configure_optimizers(self):

        if self.transformer_encoder_params is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.Adam(
                [
                    {'params': self.parameters()},
                    {'params': self.transformer_encoder_params, 'lr': 5e-5}
                ], lr=self.lr
            )

        return {
            'optimizer': optimizer,
            'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                       patience=5,
                                                                       min_lr=1e-7,
                                                                       verbose=True),
            'monitor': 'train_loss'
        }


class ChurnPredictMLP(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size1,
                 hidden_size2
                 ):
        super().__init__()
        self.output_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            # nn.BatchNorm1d(hidden_size1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size1, hidden_size2),
            # nn.BatchNorm1d(hidden_size2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.output_layer(x)


class ChurnPredictRnn(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 ):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        output, _ = self.rnn(x)
        output_lens = torch.tensor([len(x) for x in output])

        # Forward
        forward_outputs = output[list(range(output.shape[0])), output_lens - 1][:, :self.hidden_size]
        # Backward
        backward_outputs = output[list(range(output.shape[0])), [0 for _ in output_lens]][:,
                           self.hidden_size:]
        # Merge
        outputs = torch.cat([forward_outputs, backward_outputs], dim=1)

        return outputs
