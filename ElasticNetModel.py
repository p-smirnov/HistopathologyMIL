import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

# from https://www.richard-stanton.com/2021/06/19/pytorch-elasticnet.html 

class ElasticLinear(L.LightningModule):
    def __init__(self, loss_fn, n_inputs: int = 1, learning_rate=0.05, l1_lambda=0.05, l2_lambda=0.05):
        super().__init__()

        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.output_layer = torch.nn.Linear(n_inputs, 1)
        self.train_log = []

    def forward(self, x):
        outputs = self.output_layer(x)
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def l1_reg(self):
        l1_norm = self.output_layer.weight.abs().sum()

        return self.l1_lambda * l1_norm

    def l2_reg(self):
        l2_norm = self.output_layer.weight.pow(2).sum()
        
        return self.l2_lambda * l2_norm

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = torch.squeeze(self(x))
        loss = self.loss_fn(y_hat, y) + self.l1_reg() + self.l2_reg()
        
        self.log("loss", loss)
        self.train_log.append(loss.detach().numpy())
        return loss



class ElasticLogistic(L.LightningModule):
    def __init__(self, n_inputs: int = 1, learning_rate=0.05, l1_lambda=0.05, l2_lambda=0.05, class_weights=torch.tensor([1])):
        super().__init__()

        self.learning_rate = learning_rate
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.output_layer = torch.nn.Linear(n_inputs, 1)
        self.train_log = []
        self.class_weights = class_weights
        self.loss = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)

    def forward(self, x):
        outputs = self.output_layer(x)
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def l1_reg(self):
        l1_norm = self.output_layer.weight.abs().sum()

        return self.l1_lambda * l1_norm

    def l2_reg(self):
        l2_norm = self.output_layer.weight.pow(2).sum()
        
        return self.l2_lambda * l2_norm

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_logit = torch.squeeze(self(x))
        loss = self.loss(y_logit, y) + self.l1_reg() + self.l2_reg()
        y_hat = torch.sigmoid(y_logit)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        precision = torch.sum(y_hat * y) / torch.sum(y_hat)
        recall = torch.sum(y_hat * y) / torch.sum(y)
        self.log('train_prec', precision, prog_bar=True)
        self.log('train_recall', recall, prog_bar=True)
        # self.train_log.append(loss.detach().numpy())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_logit = torch.squeeze(self(x))
        loss = self.loss(y_logit, y) + self.l1_reg() + self.l2_reg()
        y_hat = torch.sigmoid(y_logit)
        precision = torch.sum(y_hat * y) / torch.sum(y_hat)
        recall = torch.sum(y_hat * y) / torch.sum(y)        
        self.log("valid_loss", loss, prog_bar=True)
        self.log('valid_prec', precision, prog_bar=True)
        self.log('valid_recall', recall, prog_bar=True)
        return loss