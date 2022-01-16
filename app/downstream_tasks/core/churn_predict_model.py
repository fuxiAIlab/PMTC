import ipdb
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import f1_score as compute_f1_score
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
    """

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


_TCN_NUM_CHANNELS = tuple([32] * 5)
_TCN_HIDDEN_SIZE = 256


class CPTCN(pl.LightningModule):
    def __init__(self,
                 seq_len,
                 handfeature_dim,
                 hidden_size=_TCN_HIDDEN_SIZE,
                 num_channels=_TCN_NUM_CHANNELS,
                 lr=1e-3,
                 use_pretrain_features=False,
                 transformer_dim=768):
        super(CPTCN, self).__init__()

        self.use_transformer_feature = use_pretrain_features

        # TODO, add use base feature
        if use_pretrain_features:
            input_dim = handfeature_dim + transformer_dim
        else:
            input_dim = handfeature_dim

        self.conv1 = nn.Conv1d(in_channels=seq_len, out_channels=hidden_size, kernel_size=1)
        self.pooling1 = nn.AvgPool1d(kernel_size=hidden_size)

        self.tcn = TemporalConvNet(num_inputs=input_dim, num_channels=num_channels)

        self.linear1 = nn.Linear(seq_len, seq_len)
        self.softmax1 = nn.Softmax(dim=-1)
        self.pooling2 = nn.MaxPool1d(kernel_size=seq_len)

        self.net_after_tcn = nn.Sequential(
            nn.BatchNorm1d(num_features=num_channels[-1] + input_dim),
            nn.Dropout(0.3),
            nn.Linear(num_channels[-1] + input_dim, 128),
            nn.BatchNorm1d(num_features=128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.Linear(64, 1)
        )

        self.lr = lr
        self.loss_func = nn.BCEWithLogitsLoss()
        # self.loss_func = F.binary_cross_entropy
        print(self)

    def forward(self, x):
        """
            x: [batch_size, seq_len, input_dim]
            seq_len：序列长度，在churn prediction里，seq_len=30，含义是最近30天内玩家的画像+行为
            input_dim = origin_portrait_dim + behavior_vector_dim # 玩家每一天的原始画像向量维度+行为向量维度
            如果玩家在该天没有画像or行为，则画像向量or行为向量取0填充向量
        """

        mean_x = torch.mean(x.view(-1, x.shape[2]), dim=0).detach()
        std_x = torch.std(x.view(-1, x.shape[2]), dim=0).detach()
        torch.isnan(std_x).any()
        x = (x - mean_x) / std_x
        x[x != x] = 0.0  # Set std-nan to 0.0

        x1 = self.conv1(x)  # [batch_size, seq_len, input_dim]
        x1 = self.pooling1(x1.transpose(1, 2)).squeeze(-1)  # [batch_size, hidden_size]

        x = x.transpose(1, 2)
        x2 = self.tcn(x)  # [batch_size, last_channel_size, seq_len]

        x2_prob = self.softmax1(self.linear1(x2))
        x2 = x2_prob * x2
        x2 = self.pooling2(x2).squeeze(-1)

        x = torch.cat([x1, x2], dim=-1)

        output = self.net_after_tcn(x)

        return output

    def forward_loss(self, batch, loss_msg):
        x, transformer_embedding, y = batch

        if self.use_transformer_feature:
            assert transformer_embedding.shape[1] != 0
            x = torch.cat([x, transformer_embedding], dim=2)

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                       patience=5,
                                                                       min_lr=1e-7,
                                                                       verbose=True),
            'monitor': 'train_loss'
        }


def main():
    from torch.utils.data import DataLoader, random_split
    import pytorch_lightning as pl
    from pytorch_lightning import seed_everything

    import sys
    sys.path.append('..')
    from game_toy_datasets import ChurnPredictToyDataset

    seed_everything(1)

    data_size = 500
    seq_len = 32
    vocab_size = 20
    input_size = 256
    transformer_dim = 768
    use_transformer_feature = True

    dataset = ChurnPredictToyDataset(data_size, seq_len, vocab_size, input_size,
                                     mask_zero_prob=0.02,
                                     use_transformer_feature=use_transformer_feature,
                                     transformer_dim=transformer_dim)
    train, val = random_split(dataset, [int(0.9 * len(dataset)), int(0.1 * len(dataset))])

    # Train config
    epoch = 10
    batch_size = 10
    trainer = pl.Trainer(max_epochs=epoch,
                         deterministic=True,
                         progress_bar_refresh_rate=1,
                         num_sanity_val_steps=2,
                         log_every_n_steps=50,
                         reload_dataloaders_every_epoch=False)

    # Init Model
    hidden_size = 16
    model = CPTCN(seq_len=seq_len,
                  input_dim=input_size,
                  hidden_size=hidden_size,
                  num_channels=[32] * 5,
                  use_pretrain_features=use_transformer_feature,
                  transformer_dim=transformer_dim)

    # Train Model
    trainer.fit(model,
                DataLoader(train, batch_size=batch_size, shuffle=True),
                DataLoader(val, batch_size=batch_size))

    # Test
    trainer.test(model, DataLoader(val, batch_size=batch_size))


if __name__ == '__main__':
    main()
