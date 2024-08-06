#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import h5py
# import wandb
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
from src.model.SimpleMILModels import Attention, MaxMIL
from src.dataloaders.DataLoaders import RetCCLFeatureLoader, RetCCLFeatureLoaderMem, RetCCLFeatureLoaderMemAllPatches


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
parser.add_argument("--run_prefix", type=str)


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

path_to_extracted_features = '/dkfz/cluster/gpu/data/OE0540/p163v/UKHD_Neuro/RetCLL_Features/' + str(args.patch_size) + '/'



import os
os.environ['HTTP_PROXY']="http://www-int.dkfz-heidelberg.de:80"
os.environ['HTTPS_PROXY']="http://www-int.dkfz-heidelberg.de:80"

slide_meta = pd.read_csv("../metadata/labels_with_new_batch.csv")
ct_scoring = pd.read_csv("../metadata/CT_3_Class_Draft.csv")



ct_scoring["txt_idat"] = ct_scoring["idat"].astype("str")
ct_scoring.index = ct_scoring.txt_idat
slide_meta.index = slide_meta.idat
ct_scoring = ct_scoring.drop("txt_idat", axis=1)
slide_meta = slide_meta.drop("idat", axis=1)
slide_annots = slide_meta.join(ct_scoring, lsuffix="l")


slide_annots['file'] = slide_annots.uuid + ".h5"


train_set = pd.read_csv(args.run_prefix + "/train_set_01.txt")
valid_set = pd.read_csv(args.run_prefix + "/valid_set_01.txt")


train_files = train_set.loc[train_set["patches"]>=args.patches_per_pat].File.tolist()
valid_files = valid_set.loc[valid_set["patches"]>=args.patches_per_pat].File.tolist()

train_annot = slide_annots.loc[[x in train_files for x in slide_annots.file]]
valid_annot = slide_annots.loc[[x in valid_files for x in slide_annots.file]]

train_labels = np.abs(1-train_annot.CT_class.factorize(sort=True)[0])
valid_labels = np.abs(1-valid_annot.CT_class.factorize(sort=True)[0])

train_file_list = [h5py.File(path_to_extracted_features + "/" + x, 'r')['feats'][:] for x in train_annot.file]
valid_file_list = [h5py.File(path_to_extracted_features + "/" + x, 'r')['feats'][:] for x in valid_annot.file]

train_data = RetCCLFeatureLoaderMem(train_file_list,train_labels, patches_per_iter=args.patches_per_pat)
valid_data = RetCCLFeatureLoaderMem(valid_file_list,valid_labels, patches_per_iter=args.patches_per_pat)

counts = np.bincount(train_labels)

pos_weight = counts[0]/counts[1]

# labels_weights = 1. / counts
# weights = labels_weights[train_labels]
# Sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
# valid_labels = valid_data.dataset.labels[valid_data.indices]

valid_counts = np.bincount(valid_labels)

print(valid_counts)


## Keeping batch size low to 
RetCCLTrain = DataLoader(train_data, batch_size=1, num_workers=9)#, sampler=Sampler)
RetCCLValid = DataLoader(valid_data, batch_size=1, num_workers=4)#, sampler=valid_Sampler)



if args.model=="Attention":
    model = Attention(2048, lr=args.lr, weight_decay=args.weight_decay, hidden_dim=args.hidden_dim, attention_dim=args.attention_dim, class_weights=torch.Tensor([pos_weight]))
elif args.model=="Max":
    model = MaxMIL(2048, lr=args.lr, weight_decay=args.weight_decay, hidden_dim=args.hidden_dim, class_weights=torch.Tensor([pos_weight]))



model = model.load_from_checkpoint(args.run_prefix + '/model.ckpt')
model.eval()


model_preds = [model(torch.tensor(x).to(model.device))[1].detach().item() for x,y in iter(RetCCLValid)]
compute_confusion_matrix(valid_labels.astype(int), np.array(model_preds).astype(int))

np.array(model_preds).tofile(args.run_prefix + "/valid_set_1_preds.csv", sep=',')



model_probs_train = [model(torch.tensor(x).to(model.device))[0].detach().item() for x,y in iter(RetCCLTrain)]
compute_confusion_matrix(valid_labels.astype(int), np.array(model_preds).astype(int))

np.array(model_probs_train).tofile(args.run_prefix + "/train_set_1_probs.csv", sep=',')


model_probs_valid = [model(torch.tensor(x).to(model.device))[0].detach().item() for x,y in iter(RetCCLValid)]

np.array(model_probs_valid).tofile(args.run_prefix + "/valid_set_1_probs.csv", sep=',')


model_attention = [model(torch.tensor(x).to(model.device))[2].detach().cpu().numpy() for x,y in iter(RetCCLValid)]


path_to_h5 = '/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/UKHD_Neuro/RetCLL_Features'

## First, just write out the per-tile attention to zarr files
for fl in np.array(valid_annot['file']):
    slidename = re.sub('\.h5', '',fl)
    print('Writing Attention Map ' + slidename)
    coords = h5py.File(path_to_h5 + "/" + fl, 'r')['coords'][:]
    outarray_root = zarr.open(args.run_prefix + "/attention_maps"+"/"+ slidename + "_per_tile_attention.zarr", mode='w') # arbitrary chunking of about 4x4 patches 
    outarray_root['coords'] = coords
    outarray_root['attn'] = np.array(model_attention[np.array(valid_annot['file']).tolist().index(fl)][0,0,:])

