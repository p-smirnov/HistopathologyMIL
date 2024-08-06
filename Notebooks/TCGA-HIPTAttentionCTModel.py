#!/usr/bin/python

import sys
sys.path.insert(1,'/home/p163v/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/software/HIPT/HIPT_4K')
import os


import numpy as np
import pandas as pd
import wandb
import zarr


import torch 
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint


from torch import optim, utils, Tensor
from SimpleMILModels import Attention, MaxMIL
from DataLoaders import HIPT256FeatureLoader


import tqdm
# from hipt_4k import HIPT_4K
from hipt_model_utils import get_vit256, eval_transforms





def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--model", type=str, default="Attention")


# Trainer arguments
parser.add_argument("--hidden_dim", type=int, default=[512], nargs="+")
parser.add_argument("--attention_dim", type=int, default=[256], nargs="+")

# Hyperparameters for the model
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--weight_decay", type=float, default=10e-4)
parser.add_argument("--max_epochs", type=int, default=200)
# parser.add_argument("--patch_size", type=int, default=1024)
parser.add_argument("--patches_per_pat", type=int, default=100)

# Parse the user inputs and defaults (returns a argparse.Namespace)
args = parser.parse_args()


## Set up Neptune logger


wandb_logger = WandbLogger(project="TCGA_HIPT_256_MIL", log_model=True, offline=False, settings=wandb.Settings(start_method="fork"))

wandb_logger.log_hyperparams(args)


# from AttentionMIL_model import Attention

path_to_extracted_features = '/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/TCGA/ffpe/HIPT_256/'



import os
os.environ['HTTP_PROXY']="http://www-int.dkfz-heidelberg.de:80"
os.environ['HTTPS_PROXY']="http://www-int.dkfz-heidelberg.de:80"

# os.environ['TENSORBOARD_BINARY'] = '/home/p163v/mambaforge/envs/marugoto/bin/tensorboard'


train_table = pd.read_csv("../metadata/TCGA_no_dup_SNP6_CT__11_1MB_10_train.csv")
valid_table = pd.read_csv("../metadata/TCGA_no_dup_SNP6_CT__11_1MB_10_valid.csv")

train_table = train_table[train_table.num_patches>=args.patches_per_pat]
valid_table = valid_table[valid_table.num_patches>=args.patches_per_pat]

train_labels = train_table.CT_Status.factorize(sort=True)[0]
train_labels = abs(train_labels-1)


valid_labels = valid_table.CT_Status.factorize(sort=True)[0]
valid_labels = abs(valid_labels-1)



# RetCCLDataset = RetCCLFeatureLoader(files, path_to_extracted_features,labels)
TrainDataset = HIPT256FeatureLoader(path_to_extracted_features, train_table.slide_name.tolist(), train_labels, args.patches_per_pat)
ValidDataset = HIPT256FeatureLoader(path_to_extracted_features, valid_table.slide_name.tolist(), valid_labels, args.patches_per_pat)


x, y = TrainDataset.__getitem__(0)


counts = np.bincount(train_labels)

pos_weight = counts[0]/counts[1]

print(counts)


# valid_labels = valid_data.dataset.labels[valid_data.indices]

# counts = np.bincount(valid_labels)
# labels_weights = 1. / counts
# weights = labels_weights[valid_labels]
# valid_Sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))


## Keeping batch size low to 
train_dataloader = DataLoader(TrainDataset, batch_size=64, num_workers=9)#, sampler=Sampler)
valid_dataloader = DataLoader(ValidDataset, batch_size=128, num_workers=4)#, sampler=valid_Sampler)


if args.model=="Attention":
    model = Attention(384, lr=args.lr, weight_decay=args.weight_decay, hidden_dim=args.hidden_dim, attention_dim=args.attention_dim, class_weights=torch.Tensor([pos_weight]))
elif args.model=="Max":
    model = MaxMIL(384, lr=args.lr, weight_decay=args.weight_decay, hidden_dim=args.hidden_dim, class_weights=torch.Tensor([pos_weight]))



checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="valid_error")


trainer = L.Trainer(max_epochs=args.max_epochs, log_every_n_steps=1, logger=wandb_logger, callbacks=[checkpoint_callback]) # limit_train_batches=100,
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

print(checkpoint_callback.best_model_path)
