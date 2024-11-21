import torch 
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch import optim, utils, Tensor
from src.transformer_from_scratch.model.transformer import Transformer



class AttentionCNVSig(L.LightningModule):
    def __init__(self, 
                input_size, 
                hidden_dim, 
                attention_dim, 
                output_dim,
                feature_extractor, 
                loss_weights,
                lr=0.0001, 
                weight_decay=50e-4,
                patch_size=256
                ):
        super().__init__()
        self.L = hidden_dim
        self.D = attention_dim
        self.K = 1
        self.lr = lr
        self.weight_decay = weight_decay
        self.output_dim = output_dim
        self.path_size = patch_size
        self.channels = 3
        self.feature_extractor_part1 = feature_extractor
        self.loss_weights = loss_weights
        self.save_hyperparameters(ignore=['feature_extractor'])

        # Features are already extracted
        # self.feature_extractor_part1 = nn.Sequential(
        #     nn.Conv2d(1, 20, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.Conv2d(20, 50, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2)
        # )
        
        # Features come in at input_size per patch
        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),            
            nn.Linear(1024, self.L),
            nn.ReLU()
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.ReLU(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, self.output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x = x.squeeze(0)
        batch_size, patches = x.shape[0:2]
        x = x.reshape([-1,self.channels,self.path_size,self.path_size])
        x = self.feature_extractor_part1(x)
        x = x.reshape([batch_size, patches, -1])
        H = self.feature_extractor_part2(x)  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 2, 1)  # KxN
        A = F.softmax(A, dim=2)  # softmax over N

        M = torch.matmul(A, H)  # KxL
        M = torch.squeeze(M)
        Y_prob = torch.squeeze(self.classifier(M),1)
        # Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, A

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        y = y.float()
        y_prob, _ = self.forward(x)
        y_prob = torch.clamp(y_prob, min=1e-8, max=1. - 1e-8)
        # loss = torch.mean(self.loss_weights * (y_prob - y)^2)  # negative log bernoulli
        loss = F.mse_loss(y_prob,y)
        #loss = self.calculate_objective(x, y)
        # x = x.view(x.size(0), -1)
        # z = self.encoder(x)
        # x_hat = self.decoder(z)
        # loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        # loss = loss.mean()
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        # self.log('train_error', error, prog_bar=True, on_step=True, on_epoch=True)
        for i in range(y.shape[1]):
            self.log('train_error_CX' + str(i+1), torch.abs(y[:,i] - y_prob[:,i]).mean(), prog_bar=False, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()
        y_prob, _ = self.forward(x)
        y_prob = torch.clamp(y_prob, min=1e-5, max=1. - 1e-5)
        # loss = -torch.sum(y*torch.log(y_prob),1)  # negative log bernoulli
        loss = F.mse_loss(y_prob,y)
        # loss = loss.mean()

        self.log("valid_loss", loss, prog_bar=True)
        # self.log('valid_error', error, prog_bar=True)
        for i in range(y.shape[1]):
            self.log('valid_error_CX' + str(i+1), torch.abs(y[:,i] - y_prob[:,i]).mean(), prog_bar=False, on_step=True, on_epoch=True)

    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=self.weight_decay)
        return optimizer

