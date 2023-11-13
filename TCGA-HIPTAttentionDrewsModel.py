#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import sys
import h5py
import wandb

import sys
sys.path.insert(1,'/home/p163v/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/software/HIPT/HIPT_4K')

from hipt_model_utils import get_vit256, eval_transforms


import torch 
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

import lightning as L
from lightning.pytorch.loggers import WandbLogger


from torch import optim, utils, Tensor
from SimpleMILModels import AttentionCNVSig
from DataLoaders import TCGAPatchLoaderImages



#### Argparser

from argparse import ArgumentParser

parser = ArgumentParser()

# Trainer arguments
parser.add_argument("--hidden_dim", type=int, default=512)

# Hyperparameters for the model
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--weight_decay", type=float, default=10e-5)
parser.add_argument("--max_epochs", type=int, default=200)
parser.add_argument("--patch_size", type=int, default=256)
parser.add_argument("--patches_per_pat", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=16)

# Parse the user inputs and defaults (returns a argparse.Namespace)
args = parser.parse_args()


## Set up WandB logger

wandb_logger = WandbLogger(project="TCGA_HIPT_256_Drews", log_model=False, offline=True, settings=wandb.Settings(start_method="fork"))

wandb_logger.log_hyperparams(args)

#########################################################
## Set up the dataset and dataloaders
#########################################################



path_to_tiles='/home/p163v/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/tiles/' + str(args.patch_size) +  '/TCGA/ffpe'

## compute the number of tiles per slide

# tiles_per_slide = [(x, len(pickle.load(open(path_to_tiles + '/' + x + '/' + x + '_tiles_list.txt', "rb")))) for x in os.listdir(path_to_tiles)]



# TilesPerPat = pd.DataFrame(tiles_per_slide, columns=["File", "Tiles Per Slide"])
TilesPerPat = pd.read_csv("../metadata/TilesPerPatTCGA_256_temp.csv")

TilesPerPat = TilesPerPat.loc[TilesPerPat["Tiles Per Slide"]>args.patches_per_pat]

slides = TilesPerPat.File.tolist()

# TilesPerPat.to_csv("../metadata/TilesPerPatTCGA_256_temp.csv", index=False)

import os
os.environ['HTTP_PROXY']="http://www-int.dkfz-heidelberg.de:80"
os.environ['HTTPS_PROXY']="http://www-int.dkfz-heidelberg.de:80"

os.environ['TENSORBOARD_BINARY'] = '/home/p163v/mambaforge/envs/marugoto/bin/tensorboard'


slide_meta = pd.read_csv("../metadata/drews_snp6_signatures_mapped.csv")

slide_meta.index = slide_meta.V1

slide_meta = slide_meta.loc[slides]

slide_meta = slide_meta.loc[~slide_meta.CX1.isna()]

slides = slide_meta.index.tolist()

labels = slide_meta[["CX1", "CX2", "CX3", "CX4", "CX5", "CX6", "CX7", "CX8", "CX9", "CX10", "CX11", "CX12", "CX13", "CX14", "CX15", "CX16", "CX17"]].values



TCGADataset = TCGAPatchLoaderImages(path_to_tiles,slides,labels, patches_per_iter = args.patches_per_pat)


# x, y = TCGADataset.__getitem__(0)


g_cpu = torch.Generator()
g_cpu.manual_seed(42)


train_data, valid_data, test_data =  torch.utils.data.random_split(TCGADataset, [0.6, 0.2, 0.2], generator=g_cpu)


# simple_dataloader = DataLoader(train_data, batch_size=10)
# prev = time.time()
# for x, y in iter(simple_dataloader):
#     cur = time.time()
#     print('Time cost {tm:.6f}s'.format(tm=cur-prev))
#     prev=cur



# train_labels = train_data.dataset.labels[train_data.indices]

# counts = np.bincount(train_labels)
# labels_weights = 1. / counts
# weights = labels_weights[train_labels]
# Sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))



# valid_labels = valid_data.dataset.labels[valid_data.indices]

# counts = np.bincount(valid_labels)
# labels_weights = 1. / counts
# weights = labels_weights[valid_labels]
# valid_Sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))


train_dataloader = DataLoader(train_data, batch_size=args.batch_size, num_workers=9)
valid_dataloader = DataLoader(valid_data, batch_size=args.batch_size, num_workers=9)


#########################################################
## Set up the model and trainer
#########################################################

pretrained_weights256 = '/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/software/HIPT/HIPT_4K/Checkpoints/vit256_small_dino.pth'
device256 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### loading pretrained ViT_256 
model256 = get_vit256(pretrained_weights=pretrained_weights256, device=device256)
model256 = model256.to(device256)
model256.eval()






model = AttentionCNVSig(384, feature_extractor = model256, 
                        lr=args.lr, output_dim=17, 
                        weight_decay=args.weight_decay, 
                        hidden_dim=args.hidden_dim, attention_dim=256,
                        loss_weights = torch.ones(17).to(device256))


trainer = L.Trainer(max_epochs=args.max_epochs, log_every_n_steps=32, logger=wandb_logger, accumulate_grad_batches=16) # limit_train_batches=100,
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)


# simple_dataloader = DataLoader(train_data, batch_size=1)
# prev = time.time()
# for x, y in iter(simple_dataloader):
#     cur = time.time()
#     print('Time cost {tm:.6f}s'.format(tm=cur-prev))
#     prev=cur

