import torch
import ipdb
import random
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import pytorch_lightning as pl
from sklearn.metrics import f1_score as compute_f1_score

MLP_HIDDEN_SIZE1 = 256
MLP_HIDDEN_SIZE2 = 256
RNN_HIDDEN_SIZE = 128


class MapPreloadModel(pl.LightningModule):
    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 mlp_hidden_size1=MLP_HIDDEN_SIZE1,
                 mlp_hidden_size2=MLP_HIDDEN_SIZE2,
                 rnn_hidden_size=RNN_HIDDEN_SIZE,
                 padding_idx=None,
                 transformer_input_size=768,
                 use_extra_feature=True,
                 rnn_num_layer=1,
                 lr=1e-3):
        super().__init__()

        if transformer_input_size is not None:
            # Add projection layer
            self.input_projection = nn.Linear(transformer_input_size, 768)
        else:
            self.input_projection = None

        self.rnn = MapPreloadRnn(vocab_size, rnn_hidden_size, rnn_hidden_size, padding_idx,
                                 num_layers=rnn_num_layer)
        rnn_output_dim = 2 * rnn_hidden_size * max_seq_len
        self.rnn_projection = nn.Linear(rnn_output_dim, 768)

        if use_extra_feature:
            mlp_input_size = self.rnn_projection.out_features + self.input_projection.out_features  # bi directional
        else:
            mlp_input_size = self.rnn_projection.out_features
        self.use_extra_feature = use_extra_feature

        self.mlp = MapPreloadMLP(vocab_size, mlp_input_size, mlp_hidden_size1, mlp_hidden_size2)
        self.loss_func = nn.CrossEntropyLoss()
        self.lr = lr
        print(self)

    def forward(self, rnn_x=None, transformer_embedding=None):
        rnn_x, x_lens = rnn_x
        rnn_embedding = self.rnn_projection(self.rnn(rnn_x, x_lens))
        if self.use_extra_feature:
            embedding = torch.cat([rnn_embedding, self.input_projection(transformer_embedding)], dim=1)
        else:
            embedding = rnn_embedding
        output = self.mlp(embedding)
        return output

    def forward_loss(self, batch, loss_msg):
        rnn_x, transformer_embedding, y = batch
        output = self(rnn_x=rnn_x, transformer_embedding=transformer_embedding)
        loss = self.loss_func(output, y.flatten())
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
        predict_labels = torch.argmax(F.softmax(output, dim=1), dim=1)

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


class MapPreloadMLP(nn.Module):
    def __init__(self,
                 vocab_size,
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
            nn.Linear(hidden_size2, vocab_size)
        )

    def forward(self, x):
        return self.output_layer(x)


class MapPreloadRnn(nn.Module):
    def __init__(self,
                 vocab_size,
                 input_size,
                 hidden_size,
                 padding_idx,
                 num_layers=1,
                 ):
        super().__init__()
        self.embeds = nn.Embedding(vocab_size, input_size, padding_idx=padding_idx)
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

    def forward(self, x, x_lens):
        x = self.embeds(x)
        packed_x = pack_padded_sequence(x, batch_first=True, lengths=x_lens)
        packed_output, _ = self.rnn(packed_x)
        unpacked_output, output_lens = pad_packed_sequence(packed_output,
                                                           batch_first=True)  # batch_size x max_seq_len x dim
        unpacked_output = unpacked_output.reshape(unpacked_output.shape[0], -1)
        outputs = unpacked_output.reshape(unpacked_output.shape[0], -1)
        return outputs


def main():
    # Use toy data to check model
    import sys
    sys.path.append('..')
    from game_toy_datasets import ToyDataset1, collate_fn_variable_len, MapPreloadDataset1
    from torch.utils.data import DataLoader, random_split
    import pytorch_lightning as pl

    # Dataset Config
    data_size = 1000
    vocab_size = 5
    max_seq_len = 64
    padding_index = 0
    transformer_input_size = 768
    use_transformer_feature = False
    dataset = MapPreloadDataset1(data_size, vocab_size, max_seq_len, padding_index,
                                 transformer_input_size,
                                 use_transformer_feature)
    train, val = random_split(dataset, [int(0.9 * len(dataset)), int(0.1 * len(dataset))])

    # Train config
    epoch = 20
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
    model = MapPreloadModel(
        vocab_size,
        max_seq_len,
        mlp_hidden_size1,
        mlp_hidden_size2,
        rnn_hidden_size=rnn_hidden_size,
        padding_idx=padding_index,
        use_base_features=True,
        use_pretrain_features=use_transformer_feature,
        transformer_input_size=transformer_input_size,
        rnn_num_layer=rnn_num_layer)

    # Train
    trainer.fit(model,
                DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_variable_len),
                DataLoader(val, batch_size=batch_size, collate_fn=collate_fn_variable_len))

    # Test
    trainer.test(model, DataLoader(val, batch_size=batch_size, collate_fn=collate_fn_variable_len))


if __name__ == '__main__':
    main()
