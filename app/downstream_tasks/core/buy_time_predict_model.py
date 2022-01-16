import torch
import ipdb
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from core.train_metric import MacroF1Metric

MLP_HIDDEN_SIZE1 = 256
MLP_HIDDEN_SIZE2 = 256
RNN_HIDDEN_SIZE = 256


class BuyTimePredictModel(pl.LightningModule):
    def __init__(self,
                 mlp_hidden_size1=MLP_HIDDEN_SIZE1,
                 mlp_hidden_size2=MLP_HIDDEN_SIZE2,
                 transformer_input_size=768,
                 lr=1e-3,
                 transformer_encoder_params=None,
                 is_finetune=False
                 ):
        super().__init__()
        self.use_pretrain_features = True
        self.transformer_encoder_params = transformer_encoder_params
        # Transformer Input projection layer
        self.input_projection = nn.Linear(transformer_input_size, 768)
        mlp_input_size = self.input_projection.out_features
        self.mlp = BuyTimePredictMLP(mlp_input_size, mlp_hidden_size1, mlp_hidden_size2)
        self.loss_func = F.binary_cross_entropy
        self.lr = lr
        self.test_metric = MacroF1Metric()
        self.val_metric = MacroF1Metric()
        self.is_select_model_by_val_metric = False
        if self.is_select_model_by_val_metric:
            assert self.val_metric is not None
        self.is_finetune = is_finetune
        print(self)

    def forward(self, X):
        # compute mean pooling for each sequence
        if self.is_finetune:
            mean_pool_X = []
            for x in X:
                # To check if the input is all zeros baselines
                if int(x[x == 0.0].shape[0]) == int(x.flatten().shape[0]):
                    mean_pool_x = torch.mean(x, dim=0)
                else:
                    # torch.is_nonzero(torch.tensor([0.]))
                    non_pad_mask = torch.sum(x, dim=1) != 0.0
                    mean_pool_x = torch.mean(x[non_pad_mask], dim=0)
                mean_pool_X.append(mean_pool_x)
            mean_pool_X = torch.stack(mean_pool_X)
        else:
            mean_pool_X = X
        embedding = self.input_projection(mean_pool_X)
        output = self.mlp(embedding)
        return output

    def forward_loss(self, batch, loss_msg):

        # transformer_embedding shape: batch_size x feature_dim
        # y shape: batch_size x 1
        (batch_seq_embedding, batch_length, _), y = batch

        output = self.forward(batch_seq_embedding)
        loss = self.loss_func(output, y.float())

        self.log(loss_msg, loss, prog_bar=True)
        return loss, output

    def training_step(self, batch, batch_idx):
        loss, _ = self.forward_loss(batch, 'train_loss')
        return loss

    def validation_step(self, batch, batch_idx, label1_thresh=0.5):
        loss, output = self.forward_loss(batch, 'val_loss')
        _, y = batch

        # compute the f1-score
        predict_labels = torch.zeros_like(output, dtype=torch.long)
        predict_labels[output > label1_thresh] = 1

        # compute labels
        true_labels = y.flatten().detach().cpu().tolist()
        predict_labels = predict_labels.flatten().detach().cpu().tolist()
        return loss, (true_labels, predict_labels)

    def test_step(self, batch, batch_idx, label1_thresh=0.5):
        loss, output = self.forward_loss(batch, 'test_loss')
        _, y = batch

        # compute the f1-score
        predict_labels = torch.zeros_like(output, dtype=torch.long)
        predict_labels[output > label1_thresh] = 1

        # compute labels
        true_labels = y.flatten().detach().cpu().tolist()
        predict_labels = predict_labels.flatten().detach().cpu().tolist()
        return loss, (true_labels, predict_labels)

    def configure_optimizers(self):

        if self.transformer_encoder_params is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            # 这里的optimizer.param_groups会分为两组，用0,1去做索引
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


class BuyTimePredictMLP(nn.Module):
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


def main():
    # Use toy data to check model
    import sys
    sys.path.append('..')
    from game_toy_datasets import ToyDataset1, collate_fn_variable_len
    from torch.utils.data import DataLoader, random_split
    import pytorch_lightning as pl

    # Dataset Config
    data_size = 1000
    vocab_size = 58
    max_seq_len = 16
    padding_index = 0
    transformer_input_size = 768
    use_transformer_feature = True
    dataset = ToyDataset1(data_size, vocab_size, max_seq_len, padding_index,
                          transformer_input_size,
                          use_transformer_feature)
    train, val = random_split(dataset, [int(0.9 * len(dataset)), int(0.1 * len(dataset))])

    # Train config
    epoch = 10
    batch_size = 64
    trainer = pl.Trainer(max_epochs=epoch,
                         deterministic=True,
                         progress_bar_refresh_rate=1,
                         num_sanity_val_steps=2,
                         log_every_n_steps=50,
                         reload_dataloaders_every_epoch=False)

    # Init model
    hidden_size = 64
    mlp_hidden_size1 = hidden_size
    mlp_hidden_size2 = hidden_size
    rnn_hidden_size = hidden_size
    rnn_num_layer = 2
    model = BuyTimePredictModel(mlp_hidden_size1,
                                mlp_hidden_size2,
                                vocab_size=vocab_size,
                                rnn_hidden_size=rnn_hidden_size,
                                padding_idx=padding_index,
                                use_base_features=True,
                                use_pretrain_features=use_transformer_feature,
                                transformer_input_size=transformer_input_size,
                                rnn_num_layer=rnn_num_layer)


if __name__ == '__main__':
    main()
