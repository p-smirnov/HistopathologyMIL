#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import h5py
import wandb


import torch 
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint


from torch import optim, utils, Tensor
from SimpleMILModels import Attention, MaxMIL
from DataLoaders import RetCCLFeatureLoader, RetCCLFeatureLoaderMem


# neptune_token = os.environ["NEPTUNE_API_TOKEN"]

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
parser.add_argument("--patch_size", type=int, default=1024)
parser.add_argument("--patches_per_pat", type=int, default=10)

# Parse the user inputs and defaults (returns a argparse.Namespace)
args = parser.parse_args()


## Set up WandB logger


wandb_logger = WandbLogger(project="UKHD_RetCLL_299_CT", log_model=True, offline=False, settings=wandb.Settings(start_method="fork"))

wandb_logger.log_hyperparams(args)

wandb_logger.experiment.config["Task"] = "GBM vs Non-GBM"


# from AttentionMIL_model import Attention

path_to_extracted_features = '/dkfz/cluster/gpu/data/OE0540/p163v/UKHD_Neuro/RetCLL_Features/' + str(args.patch_size) + '/'



import os
os.environ['HTTP_PROXY']="http://www-int.dkfz-heidelberg.de:80"
os.environ['HTTPS_PROXY']="http://www-int.dkfz-heidelberg.de:80"

os.environ['TENSORBOARD_BINARY'] = '/home/p163v/mambaforge/envs/marugoto/bin/tensorboard'


slide_meta = pd.read_csv("../metadata/slides_FS_anno.csv")
ct_scoring = pd.read_csv("../metadata/CT_3_Class_Draft.csv")



ct_scoring["txt_idat"] = ct_scoring["idat"].astype("str")
ct_scoring.index = ct_scoring.txt_idat
slide_meta.index = slide_meta.txt_idat
ct_scoring = ct_scoring.drop("txt_idat", axis=1)
slide_meta = slide_meta.drop("txt_idat", axis=1)
slide_annots = slide_meta.join(ct_scoring, lsuffix="l")


# myx = [x in ["Chromothripsis", "No Chromothripsis"] for x in slide_annots.CT_class]

# slide_annots = slide_annots.loc[myx]
slide_names = slide_annots.uuid

# slide_names
slide_annots.CT_class


# Load the data

files = [x + ".h5" for x in slide_names]

len(files)


myx = [os.path.exists(path_to_extracted_features + "/" + x) for x in files]
files = np.array(files)[myx]
len(files)


TilesPerPat = pd.read_csv("../metadata/TilesPerPat_" + str(args.patch_size) + ".csv")
filestokeep = TilesPerPat.loc[TilesPerPat["Tiles Per Slide"]>=args.patches_per_pat].File.tolist()

myx2 = [x in filestokeep for x in files]

files = files[myx2]
len(files)


labels = (slide_annots.family == 'glioblastoma').factorize()[0][myx][myx2]
labels = abs(labels-1) ## Make the positive class GBM

file_list = [h5py.File(path_to_extracted_features + "/" + x, 'r')['feats'][:] for x in files]

# RetCCLDataset = RetCCLFeatureLoader(files, path_to_extracted_features,labels)
RetCCLDataset = RetCCLFeatureLoaderMem(file_list,labels, patches_per_iter=args.patches_per_pat)


x, y = RetCCLDataset.__getitem__(0)


g_cpu = torch.Generator()
g_cpu.manual_seed(42)


train_data, valid_data, test_data =  torch.utils.data.random_split(RetCCLDataset, [0.6, 0.2, 0.2], generator=g_cpu)



train_labels = train_data.dataset.labels[train_data.indices]

counts = np.bincount(train_labels)

pos_weight = counts[0]/counts[1]

# labels_weights = 1. / counts
# weights = labels_weights[train_labels]
# Sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))



# valid_labels = valid_data.dataset.labels[valid_data.indices]

# counts = np.bincount(valid_labels)
# labels_weights = 1. / counts
# weights = labels_weights[valid_labels]
# valid_Sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))



train_dataloader = DataLoader(train_data, batch_size=128, num_workers=9)#, sampler=Sampler)
valid_dataloader = DataLoader(valid_data, batch_size=128, num_workers=4)#, sampler=valid_Sampler)


if args.model=="Attention":
    model = Attention(2048, lr=args.lr, weight_decay=args.weight_decay, hidden_dim=args.hidden_dim, attention_dim=args.attention_dim, class_weights=torch.Tensor([pos_weight]))
elif args.model=="Max":
    model = MaxMIL(2048, lr=args.lr, weight_decay=args.weight_decay, hidden_dim=args.hidden_dim, class_weights=torch.Tensor([pos_weight]))



checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="valid_loss")


trainer = L.Trainer(max_epochs=args.max_epochs, log_every_n_steps=1, logger=wandb_logger, callbacks=[checkpoint_callback]) # limit_train_batches=100,
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

print(checkpoint_callback.best_model_path)
