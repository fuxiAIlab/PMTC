import torch
import ipdb
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from core.train_metric import NDCGMetric

MLP_HIDDEN_SIZE1 = 256
MLP_HIDDEN_SIZE2 = 256
RNN_HIDDEN_SIZE = 256


class UnionRecommendModel(pl.LightningModule):
    def __init__(self,
                 mlp_hidden_size1=MLP_HIDDEN_SIZE1,
                 mlp_hidden_size2=MLP_HIDDEN_SIZE2,
                 transformer_input_size=768,
                 lr=1e-3,
                 transformer_encoder_params=None,
                 union_transformer_encoder_params=None,
                 is_finetune=False,
                 ):
        super().__init__()
        self.use_pretrain_features = True
        self.transformer_encoder_params = transformer_encoder_params
        self.union_transformer_encoder_params = union_transformer_encoder_params
        # Transformer Input projection layer
        self.input_projection = nn.Linear(transformer_input_size * 2, 768)
        mlp_input_size = self.input_projection.out_features
        self.mlp = UnionRecommendMLP(mlp_input_size, mlp_hidden_size1, mlp_hidden_size2)
        self.loss_func = F.binary_cross_entropy
        self.lr = lr
        self.test_metric = NDCGMetric()
        self.val_metric = NDCGMetric()
        self.is_select_model_by_val_metric = True
        self.is_finetune = is_finetune
        print(self)

    def forward_loss(self, batch, loss_msg):
        # transformer_embedding shape: batch_size x feature_dim
        # y shape: batch_size x 1
        transformer_embedding, y = batch
        embedding = self.input_projection(transformer_embedding)
        output = self.mlp(embedding)
        loss = self.loss_func(output, y.float())

        self.log(loss_msg, loss, prog_bar=True)
        return loss, output

    def backward(self, loss, optimizer, optimizer_idx):
        if self.is_finetune:
            # do a custom way of backward
            loss.backward(retain_graph=True)
        else:
            loss.backward()

    def training_step(self, batch, batch_idx):
        *x_y_batch, _ = batch
        loss, _ = self.forward_loss(x_y_batch, 'train_loss')
        return loss

    def validation_step(self, batch, batch_idx):
        if self.is_select_model_by_val_metric:
            *x_y_batch, role_id_ds = batch
            loss, output_prob = self.forward_loss(x_y_batch, 'val_loss')
            _, y = x_y_batch
            true_prob = y.to(torch.float)
            random_prob = torch.rand((y.shape[0], 1))
            return loss, (role_id_ds, true_prob, output_prob, random_prob)
        else:
            *x_y_batch, _ = batch
            loss, _ = self.forward_loss(x_y_batch, 'val_loss')
            return loss

    def test_step(self, batch, batch_idx, label1_thresh=0.5):
        *x_y_batch, role_id_ds = batch
        loss, output_prob = self.forward_loss(x_y_batch, 'test_loss')
        _, y = x_y_batch
        true_prob = y.to(torch.float)
        random_prob = torch.rand((y.shape[0], 1))
        return loss, (role_id_ds, true_prob, output_prob, random_prob)

    def configure_optimizers(self):

        params_list = [{'params': self.parameters()}]
        if self.transformer_encoder_params is not None:
            params_list.append({'params': self.transformer_encoder_params, 'lr': 5e-5})
        if self.union_transformer_encoder_params is not None:
            params_list.append({'params': self.union_transformer_encoder_params, 'lr': 5e-5})
        optimizer = torch.optim.Adam(params_list, lr=self.lr)

        return {
            'optimizer': optimizer,
            'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                       patience=5,
                                                                       min_lr=1e-7,
                                                                       verbose=True),
            'monitor': 'train_loss'
        }


class UnionRecommendMLP(nn.Module):
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
    pass


if __name__ == '__main__':
    main()
