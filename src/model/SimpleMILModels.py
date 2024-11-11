import torch 
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch import optim, utils, Tensor
from src.transformer_from_scratch.model.transformer import Transformer

class MaxMIL(L.LightningModule):
    def __init__(self, input_size, hidden_dim,  lr=0.0001, weight_decay=50e-4, class_weights=Tensor([1.0])):
        super().__init__()
        self.L = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.class_weights = class_weights
        self.loss = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
        self.save_hyperparameters()

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
        layer_dims = [input_size] + self.L
        layer_list = []
        for i in range(len(self.L)):
            layer_list.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            layer_list.append(nn.ReLU())
        self.feature_extractor_part2 = nn.Sequential(*layer_list)


        self.classifier = nn.Sequential(
            nn.Linear(layer_dims[-1], 1),
            # nn.Sigmoid()
        )


    def forward(self, x):
        # x = x.squeeze(0)

        # H = self.feature_extractor_part1(x)
        H = self.feature_extractor_part2(x)  # NxL

        Y_prob = torch.squeeze(torch.max(self.classifier(H),1)[0])
        Y_hat = torch.ge(torch.sigmoid(Y_prob), 0.5).float()

        return Y_prob, Y_hat, H

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        y = y.float()
        y_prob, y_hat, _ = self.forward(x)
        # y_prob = torch.clamp(y_prob, min=1e-5, max=1. - 1e-5)
        loss = self.loss(y_prob,y)
        # loss = -1. * (y * torch.log(y_prob) + (1. - y) * torch.log(1. - y_prob))  # negative log bernoulli
        error = 1. - y_hat.eq(y).float().mean()
        #loss = self.calculate_objective(x, y)
        # x = x.view(x.size(0), -1)
        # z = self.encoder(x)
        # x_hat = self.decoder(z)
        # loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        # loss = loss.mean()
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_error', error, prog_bar=True, on_step=False, on_epoch=True)
        precision = torch.sum(y_hat * y) / torch.sum(y_hat)
        recall = torch.sum(y_hat * y) / torch.sum(y)
        self.log('train_prec', precision, prog_bar=True)
        self.log('train_recall', recall, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()
        y_prob, y_hat, _ = self.forward(x)
        # y_prob = torch.clamp(y_prob, min=1e-5, max=1. - 1e-5)
        loss = self.loss(y_prob,y)
        # loss = -1. * (y * torch.log(y_prob) + (1. - y) * torch.log(1. - y_prob))  # negative log bernoulli
        error = 1. - y_hat.eq(y).float().mean()
        precision = torch.sum(y_hat * y) / torch.sum(y_hat)
        recall = torch.sum(y_hat * y) / torch.sum(y)
        # loss = loss.mean()
        self.log("valid_loss", loss, prog_bar=True)
        self.log('valid_error', error, prog_bar=True)
        self.log('valid_prec', precision, prog_bar=True)
        self.log('valid_recall', recall, prog_bar=True)
        
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=self.weight_decay)
        return optimizer



class Attention(L.LightningModule):
    def __init__(self, input_size, hidden_dim, attention_dim, lr=0.0001, weight_decay=50e-4, class_weights=Tensor([1.0])):
        super().__init__()
        self.L = hidden_dim
        self.D = attention_dim
        self.K = 1 # NUm heads?
        self.lr = lr
        self.weight_decay = weight_decay
        self.class_weights = class_weights
        self.loss = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
        self.save_hyperparameters()

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
        layer_dims = [input_size] + self.L
        layer_list = []
        for i in range(len(self.L)):
            layer_list.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            layer_list.append(nn.ReLU())
        self.feature_extractor_part2 = nn.Sequential(*layer_list)

        # self.feature_extractor_part2 = nn.Sequential(
        #     nn.Linear(input_size, 1024),
        #     nn.ReLU(),            
        #     nn.Linear(1024, self.L),
        #     nn.ReLU()
        # )

        attention_dims = [input_size] + self.D

        attention_list = []
        for i in range(len(self.D)):
            attention_list.append(nn.Linear(attention_dims[i], attention_dims[i+1]))
            attention_list.append(nn.ReLU())
        
        attention_list.append(nn.Linear(attention_dims[-1], self.K))
        self.attention = nn.Sequential(*attention_list)

        # self.attention = nn.Sequential(
        #     nn.Linear(self.L, self.D),
        #     nn.ReLU(),
        #     nn.Linear(self.D, self.K)
        # )

        self.classifier = nn.Sequential(
            nn.Linear(layer_dims[-1]*self.K, 1),
            # nn.Sigmoid()
        )

    def forward(self, x, **kwargs):
        # x = x.squeeze(0)
        mask = kwargs.get('mask', None)
        # H = self.feature_extractor_part1(x)
        H = self.feature_extractor_part2(x)  # NxL


        # A = self.attention(x)  # NxK
        # A = torch.transpose(A, 2, 1)  # KxN
        # A = F.softmax(A, dim=2)  # softmax over N
        A = self.attention_forward(x, mask)

        M = torch.matmul(A, H)  # KxL
        M = torch.squeeze(M, 1)
        Y_prob = torch.squeeze(self.classifier(M),1)
        Y_hat = torch.ge(torch.sigmoid(Y_prob), 0.5).float()

        return Y_prob, Y_hat, A
    
    def attention_forward(self, x, mask = None):
        if mask is None:
            mask = torch.zeros(x.shape[0], x.shape[1]).to(x.device)
        A = self.attention(x)  # NxK
        A = A + mask.unsqueeze(2)
        A = torch.transpose(A, 2, 1)  # KxN
        A = F.softmax(A, dim=2)  # softmax over N
        return A

    def score_forward(self, x):
        H = self.feature_extractor_part2(x)  # NxL
        return self.classifier(H)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y, mask = batch
        y = y.float()
        y_prob, y_hat, _ = self.forward(x, mask=mask)
        # y_prob = torch.clamp(y_prob, min=1e-5, max=1. - 1e-5)
        loss = self.loss(y_prob,y)
        # loss = -1. * (y * torch.log(y_prob) + (1. - y) * torch.log(1. - y_prob))  # negative log bernoulli
        error = 1. - y_hat.eq(y).float().mean()
        #loss = self.calculate_objective(x, y)
        # x = x.view(x.size(0), -1)
        # z = self.encoder(x)
        # x_hat = self.decoder(z)
        # loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        # loss = loss.mean()
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_error', error, prog_bar=True, on_step=False, on_epoch=True)
        precision = torch.sum(y_hat * y) / (torch.sum(y_hat)+1e-8)
        recall = torch.sum(y_hat * y) / (torch.sum(y)+1e-8)
        if precision.isfinite().item(): 
            self.log('train_prec', precision, prog_bar=True)
        if recall.isfinite().item():
            self.log('train_recall', recall, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        y = y.float()
        y_prob, y_hat, _ = self.forward(x, mask=mask)
        # y_prob = torch.clamp(y_prob, min=1e-5, max=1. - 1e-5)
        loss = self.loss(y_prob,y)
        # loss = -1. * (y * torch.log(y_prob) + (1. - y) * torch.log(1. - y_prob))  # negative log bernoulli
        error = 1. - y_hat.eq(y).float().mean()
        precision = torch.sum(y_hat * y) / (torch.sum(y_hat)+1e-8)
        recall = torch.sum(y_hat * y) / (torch.sum(y)+1e-8)
        # loss = loss.mean()
        self.log("valid_loss", loss, prog_bar=True)
        self.log('valid_error', error, prog_bar=True)
        if precision.isfinite().item(): 
            self.log('valid_prec', precision, prog_bar=True)
        if recall.isfinite().item():
            self.log('valid_recall', recall, prog_bar=True)
        if recall.isfinite().item() and precision.isfinite().item():
            f1score = 2 * precision * recall / (precision + recall + 1e-8)
            self.log('valid_f1', f1score, prog_bar=True)


    def test_step(self, batch, batch_idx):
        x, y, mask = batch
        y = y.float()
        y_prob, y_hat, _ = self.forward(x, mask=mask)
        # y_prob = torch.clamp(y_prob, min=1e-5, max=1. - 1e-5)
        loss = self.loss(y_prob,y)
        # loss = -1. * (y * torch.log(y_prob) + (1. - y) * torch.log(1. - y_prob))  # negative log bernoulli
        error = 1. - y_hat.eq(y).float().mean()
        precision = torch.sum(y_hat * y) / (torch.sum(y_hat)+1e-8)
        recall = torch.sum(y_hat * y) / (torch.sum(y)+1e-8)
        # loss = loss.mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log('test_error', error, prog_bar=True)
        if precision.isfinite().item(): 
            self.log('test_prec', precision, prog_bar=True)
        if recall.isfinite().item():
            self.log('test_recall', recall, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=self.weight_decay)
        return optimizer


class ResConnection(nn.Module):
    # num_features: the number of inputs for the Residual block
    # num_squeeze: the number of middle neurons for the Residual block
    def __init__(self, num_features,num_squeeze):
        super().__init__()
        self.num_features = num_features
        self.num_squeeze = num_squeeze

        self.linear1 = nn.Linear(num_features, num_squeeze, bias=False)
        self.linear2 = nn.Linear(num_squeeze, num_features, bias=False)
        self.bn1 = nn.BatchNorm1d(num_squeeze)
        self.bn2 = nn.BatchNorm1d(num_features)

    def forward(self, X):
        x = X
        num_patches = X.shape[1]
        x = F.relu(self.bn1(self.linear1(x).reshape((-1,self.num_squeeze))).reshape((-1,num_patches,self.num_squeeze)))
        x = F.relu(self.bn2(self.linear2(x).reshape((-1,self.num_features))).reshape((-1,num_patches,self.num_features)))
        return x + X

class AttentionResNet(L.LightningModule):
    def __init__(self, input_size, hidden_dim, attention_dim, lr=0.0001, weight_decay=50e-4, class_weights=Tensor([1.0])):
        super().__init__()
        self.L = hidden_dim
        self.D = attention_dim
        self.K = 1 # NUm heads?
        self.lr = lr
        self.weight_decay = weight_decay
        self.class_weights = class_weights
        self.loss = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
        self.save_hyperparameters()
        
        # Features come in at input_size per patch
        layer_dims = self.L
        layer_list = []
        for i in range(len(self.L)):
            layer_list.append(ResConnection(input_size,layer_dims[i]))
        self.feature_extractor_part2 = nn.Sequential(*layer_list)

        # self.feature_extractor_part2 = nn.Sequential(
        #     nn.Linear(input_size, 1024),
        #     nn.ReLU(),            
        #     nn.Linear(1024, self.L),
        #     nn.ReLU()
        # )

        attention_dims = [input_size] + self.D

        attention_list = []
        for i in range(len(self.D)):
            attention_list.append(nn.Linear(attention_dims[i], attention_dims[i+1]))
            attention_list.append(nn.ReLU())
        
        attention_list.append(nn.Linear(attention_dims[-1], self.K))
        self.attention = nn.Sequential(*attention_list)

        self.classifier = nn.Sequential(
            nn.Linear(input_size*self.K, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        # x = x.squeeze(0)

        # H = self.feature_extractor_part1(x)
        H = self.feature_extractor_part2(x)  # NxL

        A = self.attention(x)  # NxK
        A = torch.transpose(A, 2, 1)  # KxN
        A = F.softmax(A, dim=2)  # softmax over N

        M = torch.matmul(A, H)  # KxL
        M = torch.squeeze(M, 1)
        Y_prob = torch.squeeze(self.classifier(M),1)
        Y_hat = torch.ge(torch.sigmoid(Y_prob), 0.5).float()

        return Y_prob, Y_hat, A

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        y = y.float()
        y_prob, y_hat, _ = self.forward(x)
        # y_prob = torch.clamp(y_prob, min=1e-5, max=1. - 1e-5)
        loss = self.loss(y_prob,y)
        # loss = -1. * (y * torch.log(y_prob) + (1. - y) * torch.log(1. - y_prob))  # negative log bernoulli
        error = 1. - y_hat.eq(y).float().mean()
        #loss = self.calculate_objective(x, y)
        # x = x.view(x.size(0), -1)
        # z = self.encoder(x)
        # x_hat = self.decoder(z)
        # loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        # loss = loss.mean()
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_error', error, prog_bar=True, on_step=False, on_epoch=True)
        precision = torch.sum(y_hat * y) / (torch.sum(y_hat)+1e-8)
        recall = torch.sum(y_hat * y) / (torch.sum(y)+1e-8)
        if precision.isfinite().item(): 
            self.log('train_prec', precision, prog_bar=True)
        if recall.isfinite().item():
            self.log('train_recall', recall, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()
        y_prob, y_hat, _ = self.forward(x)
        # y_prob = torch.clamp(y_prob, min=1e-5, max=1. - 1e-5)
        loss = self.loss(y_prob,y)
        # loss = -1. * (y * torch.log(y_prob) + (1. - y) * torch.log(1. - y_prob))  # negative log bernoulli
        error = 1. - y_hat.eq(y).float().mean()
        precision = torch.sum(y_hat * y) / (torch.sum(y_hat)+1e-8)
        recall = torch.sum(y_hat * y) / (torch.sum(y)+1e-8)
        # loss = loss.mean()
        self.log("valid_loss", loss, prog_bar=True)
        self.log('valid_error', error, prog_bar=True)
        if precision.isfinite().item(): 
            self.log('valid_prec', precision, prog_bar=True)
        if recall.isfinite().item():
            self.log('valid_recall', recall, prog_bar=True)

    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=self.weight_decay)
        return optimizer




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



class TransformerMIL(L.LightningModule):
    def __init__(self, input_size, hidden_dim, n_heads, lr=0.0001, weight_decay=50e-4, class_weights=Tensor([1.0]), position_encoding=None):
        super().__init__()
        assert all([x==hidden_dim[0] for x in hidden_dim])
        self.input_size = input_size
        self.embedding_dim = hidden_dim[0]
        self.n_layers = len(hidden_dim)
        self.n_heads = n_heads
        self.lr = lr
        self.weight_decay = weight_decay
        self.class_weights = class_weights
        self.loss = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
        self.position_encoding = position_encoding ## passed in as a parameter


        self.transformer_config = {
            "n_trans_blocks": self.n_layers,
            "input_size": self.input_size,
            "tb": {
                "n_heads": self.n_heads,
                "embedding_size": self.embedding_dim,
                "mlp_dropout": 0.2,
                "mlp_bias": True,
                "mha_dropout": 0.2,
                "mha_bias": False
            },
            "cls": True
        }
        
        self.save_hyperparameters()
        self.model = Transformer(False, self.transformer_config)


    def forward(self, x, **kwargs):
        mask = kwargs.get('mask', None)
        Y_prob = self.model(x, None).squeeze()

        Y_hat = torch.ge(torch.sigmoid(Y_prob), 0.5).float()

        return Y_prob, Y_hat
    ## TODO: implement attn rollout    
    # def attention_forward(self, x):
    #     A = self.attention(x)  # NxK
    #     A = torch.transpose(A, 2, 1)  # KxN
    #     A = F.softmax(A, dim=2)  # softmax over N
    #     return A

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y, mask = batch
        y = y.float()
        y_prob, y_hat = self.forward(x)
        # y_prob = torch.clamp(y_prob, min=1e-5, max=1. - 1e-5)
        loss = self.loss(y_prob,y)
        # loss = -1. * (y * torch.log(y_prob) + (1. - y) * torch.log(1. - y_prob))  # negative log bernoulli
        error = 1. - y_hat.eq(y).float().mean()
        #loss = self.calculate_objective(x, y)
        # x = x.view(x.size(0), -1)
        # z = self.encoder(x)
        # x_hat = self.decoder(z)
        # loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        # loss = loss.mean()
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_error', error, prog_bar=True, on_step=False, on_epoch=True)
        precision = torch.sum(y_hat * y) / (torch.sum(y_hat)+1e-8)
        recall = torch.sum(y_hat * y) / (torch.sum(y)+1e-8)
        if precision.isfinite().item(): 
            self.log('train_prec', precision, prog_bar=True)
        if recall.isfinite().item():
            self.log('train_recall', recall, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        y = y.float()
        y_prob, y_hat = self.forward(x)
        # y_prob = torch.clamp(y_prob, min=1e-5, max=1. - 1e-5)
        loss = self.loss(y_prob,y)
        # loss = -1. * (y * torch.log(y_prob) + (1. - y) * torch.log(1. - y_prob))  # negative log bernoulli
        error = 1. - y_hat.eq(y).float().mean()
        precision = torch.sum(y_hat * y) / (torch.sum(y_hat)+1e-8)
        recall = torch.sum(y_hat * y) / (torch.sum(y)+1e-8)
        # loss = loss.mean()
        self.log("valid_loss", loss, prog_bar=True)
        self.log('valid_error', error, prog_bar=True)
        if precision.isfinite().item(): 
            self.log('valid_prec', precision, prog_bar=True)
        if recall.isfinite().item():
            self.log('valid_recall', recall, prog_bar=True)
        if recall.isfinite().item() and precision.isfinite().item():
            f1score = 2 * precision * recall / (precision + recall + 1e-8)
            self.log('valid_f1', f1score, prog_bar=True)


    def test_step(self, batch, batch_idx):
        x, y, mask = batch
        y = y.float()
        y_prob, y_hat = self.forward(x)
        # y_prob = torch.clamp(y_prob, min=1e-5, max=1. - 1e-5)
        loss = self.loss(y_prob,y)
        # loss = -1. * (y * torch.log(y_prob) + (1. - y) * torch.log(1. - y_prob))  # negative log bernoulli
        error = 1. - y_hat.eq(y).float().mean()
        precision = torch.sum(y_hat * y) / (torch.sum(y_hat)+1e-8)
        recall = torch.sum(y_hat * y) / (torch.sum(y)+1e-8)
        # loss = loss.mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log('test_error', error, prog_bar=True)
        if precision.isfinite().item(): 
            self.log('test_prec', precision, prog_bar=True)
        if recall.isfinite().item():
            self.log('test_recall', recall, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=self.weight_decay)
        return optimizer