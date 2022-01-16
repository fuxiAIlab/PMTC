import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score as compute_f1_score


class GruBase1(pl.LightningModule):

    def __init__(self, hidden_size, vocab_size, max_seq_len, lr):
        super().__init__()
        self.input_embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, 1, batch_first=True, bidirectional=False)
        self.cls = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size))
        self.lr = lr

    def forward(self, x):
        x_embeddings = self.input_embedding(x)
        output, hn = self.rnn(x_embeddings)
        output = output[:, -1, :]
        output = self.cls(output)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def training_end(self, outputs):
        self.train_losses = outputs

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def validation_end(self, outputs):
        self.val_losses = outputs

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y)

        # compute the f1-score
        labels_predict = torch.argmax(output, dim=1)
        test_metric_value = compute_f1_score(y, labels_predict, average='macro')
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_metric_value', test_metric_value, prog_bar=True)
        return loss, test_metric_value

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
