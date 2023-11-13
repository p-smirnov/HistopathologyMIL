#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import h5py
import wandb
import zarr
import re

import torch 
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint


from torch import optim, utils, Tensor
from SimpleMILModels import Attention, MaxMIL
from DataLoaders import RetCCLFeatureLoader, RetCCLFeatureLoaderMem, RetCCLFeatureLoaderMemAllPatches


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
parser.add_argument("--patch_size", type=int, default=299)
parser.add_argument("--patches_per_pat", type=int, default=10)


# Parse the user inputs and defaults (returns a argparse.Namespace)
args = parser.parse_args()

if args.patch_size == 299:
  real_patch_size = 384
else:
  real_patch_size = args.patch_size

## https://stackoverflow.com/questions/2148543/how-to-write-a-confusion-matrix
def compute_confusion_matrix(true, pred):
  '''Computes a confusion matrix using numpy for two np.arrays
  true and pred.
  Results are identical (and similar in computation time) to: 
  "from sklearn.metrics import confusion_matrix"
  However, this function avoids the dependency on sklearn.'''
  K = len(np.unique(true)) # Number of classes 
  result = np.zeros((K, K))
  for i in range(len(true)):
    result[true[i]][pred[i]] += 1
  return result


## Set up Neptune logger


# wandb_logger = WandbLogger(project="UKHD_RetCLL_299_CT", log_model=True, offline=False, settings=wandb.Settings(start_method="fork"))

# wandb_logger.log_hyperparams(args)


# from AttentionMIL_model import Attention

path_to_extracted_features = '/dkfz/cluster/gpu/data/OE0540/p163v/UKHD_Neuro/RetCLL_Features/' + str(args.patch_size) + '/'



import os
os.environ['HTTP_PROXY']="http://www-int.dkfz-heidelberg.de:80"
os.environ['HTTPS_PROXY']="http://www-int.dkfz-heidelberg.de:80"

# os.environ['TENSORBOARD_BINARY'] = '/home/p163v/mambaforge/envs/marugoto/bin/tensorboard'


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

train_data = pd.read_csv("../metadata/train_set_1.txt", header=None)
valid_data = pd.read_csv("../metadata/valid_set_1.txt", header=None)

slide_annots.index = slide_annots.uuid + ".h5"

train_data.index = train_data[0]
valid_data.index = valid_data[0]

train_data = train_data.join(slide_annots)
valid_data = valid_data.join(slide_annots)
train_labels = train_data.CT_class.factorize(sort=True)[0]
train_labels = abs(train_labels -1)
valid_labels = valid_data.CT_class.factorize(sort=True)[0]
valid_labels = abs(valid_labels -1)

# Load the data

train_data_list = [h5py.File(path_to_extracted_features + "/" + x, 'r')['feats'][:] for x in np.array(train_data[0])]
valid_data_list = [h5py.File(path_to_extracted_features + "/" + x, 'r')['feats'][:] for x in np.array(valid_data[0])]


RetCCLTrain = RetCCLFeatureLoaderMemAllPatches(train_data_list,train_labels, patches_per_iter=args.patches_per_pat)
RetCCLValid = RetCCLFeatureLoaderMemAllPatches(valid_data_list,valid_labels, patches_per_iter=args.patches_per_pat)


x, y = RetCCLTrain.__getitem__(0)


counts = np.bincount(train_labels)

pos_weight = counts[0]/counts[1]

# labels_weights = 1. / counts
# weights = labels_weights[train_labels]
# Sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

valid_counts = np.bincount(valid_labels)

print(valid_counts)
# valid_files = [files[i] for i in valid_data.indices]
# train_files = [files[i] for i in train_data.indices]
# with open('../metadata/valid_set_1.txt', 'w') as f:  
#     for fn in valid_files: 
#         f.write(fn + "\n")

# with open('../metadata/train_set_1.txt', 'w') as f:  
#     for fn in train_files: 
#         f.write(fn + "\n")



# valid_labels = valid_data.dataset.labels[valid_data.indices]

# counts = np.bincount(valid_labels)
# labels_weights = 1. / counts
# weights = labels_weights[valid_labels]
# valid_Sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))


## Keeping batch size low to 
# train_dataloader = DataLoader(RetCCLTrain, batch_size=64, num_workers=9)#, sampler=Sampler)
# valid_data = DataLoader(RetCLLValid, batch_size=128, num_workers=4)#, sampler=valid_Sampler)


if args.model=="Attention":
    model = Attention(2048, lr=args.lr, weight_decay=args.weight_decay, hidden_dim=args.hidden_dim, attention_dim=args.attention_dim, class_weights=torch.Tensor([pos_weight]))
elif args.model=="Max":
    model = MaxMIL(2048, lr=args.lr, weight_decay=args.weight_decay, hidden_dim=args.hidden_dim, class_weights=torch.Tensor([pos_weight]))



model = model.load_from_checkpoint('./UKHD_RetCLL_299_CT/0odhvfgy/checkpoints/epoch=37-step=532.ckpt')
model.eval()


model_preds = [model(torch.unsqueeze(torch.tensor(x).to(model.device),0))[1].detach().item() for x,y in iter(RetCCLValid)]
compute_confusion_matrix(valid_labels.astype(int), np.array(model_preds).astype(int))

np.array(model_preds).tofile('../metadata/valid_set_1_preds.csv', sep=',')



model_probs_train = [model(torch.unsqueeze(torch.tensor(x).to(model.device),0))[0].detach().item() for x,y in iter(RetCCLTrain)]
compute_confusion_matrix(valid_labels.astype(int), np.array(model_preds).astype(int))

np.array(model_probs_train).tofile('../metadata/train_set_1_probs.csv', sep=',')


model_probs_valid = [model(torch.unsqueeze(torch.tensor(x).to(model.device),0))[0].detach().item() for x,y in iter(RetCCLValid)]

np.array(model_probs_valid).tofile('../metadata/valid_set_1_probs.csv', sep=',')


model_attention = [model(torch.unsqueeze(torch.tensor(x).to(model.device),0))[2].detach().cpu().numpy() for x,y in iter(RetCCLValid)]


path_to_h5 = '/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/UKHD_Neuro/RetCLL_Features'

## First, just write out the per-tile attention to zarr files
for fl in list(valid_data[0]):
    slidename = re.sub('\.h5', '',fl)
    print('Writing Attention Map ' + slidename)
    coords = h5py.File(path_to_h5 + "/" + fl, 'r')['coords'][:]
    outarray_root = zarr.open("../metadata/attention_maps"+"/"+ slidename + "_per_tile_attention.zarr", mode='w') # arbitrary chunking of about 4x4 patches 
    outarray_root['coords'] = coords
    outarray_root['attn'] = np.array(model_attention[list(valid_data[0]).index(fl)][0,0,:])

