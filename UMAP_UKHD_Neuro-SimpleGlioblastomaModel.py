#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import sys
import h5py
import re
import sklearn


import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch import optim, utils, Tensor



# from AttentionMIL_model import Attention

path_to_extracted_features = '/dkfz/cluster/gpu/data/OE0540/p163v/UKHD_Neuro/RetCLL_Features/'




# In[2]:


class Attention(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.L = 512
        self.D = 256
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
            nn.Linear(2048, 1024),
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
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = x.squeeze(0)

        # H = self.feature_extractor_part1(x)
        H = self.feature_extractor_part2(x)  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 2, 1)  # KxN
        A = F.softmax(A, dim=2)  # softmax over N

        M = torch.matmul(A, H)  # KxL
        M = torch.squeeze(M)
        Y_prob = torch.squeeze(self.classifier(M),1)
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

        return neg_log_likelihood.mean()
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        y = y.float()
        y_prob, y_hat, _ = self.forward(x)
        y_prob = torch.clamp(y_prob, min=1e-5, max=1. - 1e-5)
        loss = -1. * (y * torch.log(y_prob) + (1. - y) * torch.log(1. - y_prob))  # negative log bernoulli
        error = 1. - y_hat.eq(y).float().mean()
        #loss = self.calculate_objective(x, y)
        # x = x.view(x.size(0), -1)
        # z = self.encoder(x)
        # x_hat = self.decoder(z)
        # loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        loss = loss.mean()
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_error', error, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0005, betas=(0.9, 0.999), weight_decay=10e-5)
        return optimizer

    


# In[3]:


class RetCCLFeatureLoader(Dataset):
    def __init__(self, slide_filenames, feature_path, labels, patches_per_iter = 500):
        assert len(labels) == len(slide_filenames)
        self.labels = labels
        self.file_paths = [feature_path + "/" +  x for x in slide_filenames]
        self.slide_names = slide_filenames
        self.num_patches = patches_per_iter

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        cur_path = self.file_paths[idx]
        label = self.labels[idx]
        features = h5py.File(cur_path, 'r')['feats']
        # features = features.reshape([1,-1,2048])
        sampled_pchs = np.random.choice(range(features.shape[0]),size=self.num_patches, replace=False)
        sampled_pchs = np.sort(sampled_pchs)
        features = features[sampled_pchs,:]
        return features.astype(np.float32), label


# In[19]:


import os
os.environ['HTTP_PROXY']="http://www-int.dkfz-heidelberg.de:80"
os.environ['HTTPS_PROXY']="http://www-int.dkfz-heidelberg.de:80"

os.environ['TENSORBOARD_BINARY'] = '/home/p163v/mambaforge/envs/marugoto/bin/tensorboard'


# In[5]:


slide_meta = pd.read_csv("../metadata/slides_FS_anno.csv")
ct_scoring = pd.read_csv("../metadata/CT_3_Class_Draft.csv")



ct_scoring["txt_idat"] = ct_scoring["idat"].astype("str")
ct_scoring.index = ct_scoring.txt_idat
slide_meta.index = slide_meta.txt_idat
ct_scoring = ct_scoring.drop("txt_idat", axis=1)
slide_meta = slide_meta.drop("txt_idat", axis=1)
slide_annots = slide_meta.join(ct_scoring, lsuffix="l")


myx = [x in ["Chromothripsis", "No Chromothripsis"] for x in slide_annots.CT_class]

slide_annots = slide_annots.loc[myx]
slide_names = slide_annots.uuid

# slide_names
slide_annots.CT_class


# In[6]:



# Load the data

files = [x + ".h5" for x in slide_names]

len(files)


# In[7]:


myx = [os.path.exists(path_to_extracted_features + "/" + x) for x in files]
files = np.array(files)[myx]
len(files)


# In[8]:


TilesPerPat = pd.read_csv("../metadata/TilesPerPat.csv")
filestokeep = TilesPerPat.loc[TilesPerPat["Tiles Per Slide"]>=500].File.tolist()

myx2 = [x in filestokeep for x in files]

files = files[myx2]
len(files)


# In[9]:


labels = (slide_annots.family == 'glioblastoma').factorize()[0][myx][myx2]


# In[10]:


RetCCLDataset = RetCCLFeatureLoader(files, path_to_extracted_features,labels)
x, y = RetCCLDataset.__getitem__(0)


# In[11]:


train_data, test_data =  torch.utils.data.random_split(RetCCLDataset, [0.8, 0.2])

train_labels = train_data.dataset.labels[train_data.indices]

counts = np.bincount(train_labels)
labels_weights = 1. / counts
weights = labels_weights[train_labels]
Sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

# In[12]:


train_dataloader = DataLoader(train_data, batch_size=64, num_workers=9, sampler=Sampler)


# In[13]:


model = Attention()





trainer = L.Trainer(max_epochs=50, log_every_n_steps=1) # limit_train_batches=100,
trainer.fit(model, train_dataloader, val_dataloader)




