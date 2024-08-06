#!/usr/bin/env python
# coding: utf-8

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
from src.model.SimpleMILModels import MaxMIL
from src.dataloaders.DataLoaders import RetCCLFeatureLoader, RetCCLFeatureLoaderMem



# from AttentionMIL_model import Attention

path_to_extracted_features = '/dkfz/cluster/gpu/data/OE0540/p163v/UKHD_Neuro/RetCLL_Features/1024'




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


myx = [x in ["Chromothripsis", "No Chromothripsis"] for x in slide_annots.CT_class]

slide_annots = slide_annots.loc[myx]
slide_names = slide_annots.uuid

# slide_names
slide_annots.CT_class


# Load the data

files = [x + ".h5" for x in slide_names]

# with h5py.File(filename, "r") as f:
#     # Print all root level object names (aka keys) 
#     # these can be group or dataset names 
#     print("Keys: %s" % f.keys())
#     # get first object name/key; may or may NOT be a group
#     a_group_key = list(f.keys())[0]

#     # get the object type for a_group_key: usually group or dataset
#     print(type(f[a_group_key])) 

#     # If a_group_key is a group name, 
#     # this gets the object names in the group and returns as a list
#     data = list(f[a_group_key])

#     # If a_group_key is a dataset name, 
#     # this gets the dataset values and returns as a list
#     data = list(f[a_group_key])
#     # preferred methods to get dataset values:
#     ds_obj = f[a_group_key]      # returns as a h5py dataset object
#     ds_arr = f[a_group_key][()]  # returns as a numpy array
len(files)


myx = [os.path.exists(path_to_extracted_features + "/" + x) for x in files]
files = np.array(files)[myx]
len(files)


TilesPerPat = pd.read_csv("../metadata/TilesPerPat_1024.csv")
filestokeep = TilesPerPat.loc[TilesPerPat["Tiles Per Slide"]>=10].File.tolist()

myx2 = [x in filestokeep for x in files]

files = files[myx2]
len(files)


labels = slide_annots.CT_class.factorize()[0][myx][myx2]
labels = abs(labels-1)


file_list = [h5py.File(path_to_extracted_features + "/" + x, 'r')['feats'][:] for x in files]

# RetCCLDataset = RetCCLFeatureLoader(files, path_to_extracted_features,labels)
RetCCLDataset = RetCCLFeatureLoaderMem(file_list,labels)


g_cpu = torch.Generator()
g_cpu.manual_seed(42)


train_data, valid_data, test_data =  torch.utils.data.random_split(RetCCLDataset, [0.6, 0.2, 0.2], generator=g_cpu)



train_labels = train_data.dataset.labels[train_data.indices]

counts = np.bincount(train_labels)
labels_weights = 1. / counts
weights = labels_weights[train_labels]
Sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))



valid_labels = valid_data.dataset.labels[valid_data.indices]

counts = np.bincount(valid_labels)
labels_weights = 1. / counts
weights = labels_weights[valid_labels]
valid_Sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))


train_dataloader = DataLoader(train_data, batch_size=128, num_workers=9, sampler=Sampler)
valid_dataloader = DataLoader(valid_data, batch_size=64, num_workers=4, sampler=valid_Sampler)

model = MaxMIL(2048, lr=args.lr, weight_decay=args.weight_decay, hidden_dim=args.hidden_dim)


trainer = L.Trainer(max_epochs=args.max_epochs, log_every_n_steps=1) # limit_train_batches=100,
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)




