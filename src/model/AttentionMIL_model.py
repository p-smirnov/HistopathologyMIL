import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch import optim, utils, Tensor


class Attention(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        # Features are already extracted
        # self.feature_extractor_part1 = nn.Sequential(
        #     nn.Conv2d(1, 20, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.Conv2d(20, 50, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2)
        # )
        
        # Features come in at 2048 per patch
        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(2048, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = x.squeeze(0)

        # H = self.feature_extractor_part1(x)
        x = 
        H = self.feature_extractor_part2(x)  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        loss = self.calculate_objective(x, y)
        # x = x.view(x.size(0), -1)
        # z = self.encoder(x)
        # x_hat = self.decoder(z)
        # loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    

# class GatedAttention(nn.Module):
#     def __init__(self):
#         super(GatedAttention, self).__init__()
#         self.L = 500
#         self.D = 128
#         self.K = 1

#         self.feature_extractor_part1 = nn.Sequential(
#             nn.Conv2d(1, 20, kernel_size=5),
#             nn.ReLU(),
#             nn.MaxPool2d(2, stride=2),
#             nn.Conv2d(20, 50, kernel_size=5),
#             nn.ReLU(),
#             nn.MaxPool2d(2, stride=2)
#         )

#         self.feature_extractor_part2 = nn.Sequential(
#             nn.Linear(50 * 4 * 4, self.L),
#             nn.ReLU(),
#         )

#         self.attention_V = nn.Sequential(
#             nn.Linear(self.L, self.D),
#             nn.Tanh()
#         )

#         self.attention_U = nn.Sequential(
#             nn.Linear(self.L, self.D),
#             nn.Sigmoid()
#         )

#         self.attention_weights = nn.Linear(self.D, self.K)

#         self.classifier = nn.Sequential(
#             nn.Linear(self.L*self.K, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = x.squeeze(0)

#         H = self.feature_extractor_part1(x)
#         H = H.view(-1, 50 * 4 * 4)
#         H = self.feature_extractor_part2(H)  # NxL

#         A_V = self.attention_V(H)  # NxD
#         A_U = self.attention_U(H)  # NxD
#         A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
#         A = torch.transpose(A, 1, 0)  # KxN
#         A = F.softmax(A, dim=1)  # softmax over N

#         M = torch.mm(A, H)  # KxL

#         Y_prob = self.classifier(M)
#         Y_hat = torch.ge(Y_prob, 0.5).float()

#         return Y_prob, Y_hat, A

#     # AUXILIARY METHODS
#     def calculate_classification_error(self, X, Y):
#         Y = Y.float()
#         _, Y_hat, _ = self.forward(X)
#         error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

#         return error, Y_hat

#     def calculate_objective(self, X, Y):
#         Y = Y.float()
#         Y_prob, _, A = self.forward(X)
#         Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
#         neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

#         return neg_log_likelihood, A
