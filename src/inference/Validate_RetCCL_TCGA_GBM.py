#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import os
import glob
import h5py
import wandb

import string
import random

import torch 
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint


from torch import optim, utils, Tensor

import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
import sklearn.linear_model


from itables import show
from src.model.SimpleMILModels import Attention, MaxMIL, AttentionResNet
from src.dataloaders.DataLoaders import RetCCLFeatureLoader, RetCCLFeatureLoaderMem


import zarr
import seaborn as sns

recompute_attn = False




def sigmoid_array(x):
    return 1 / (1 + np.exp(-x))




import os
os.environ['HTTP_PROXY']="http://www-int.dkfz-heidelberg.de:80"
os.environ['HTTPS_PROXY']="http://www-int.dkfz-heidelberg.de:80"




group = "Q24L5M"




api = wandb.Api()




runs = api.runs(path="psmirnov/UKHD_RetCLL_299_CT", filters={"group": group})




runs


# Load in the features (not too much memory needed)



path_to_extracted_features = '/home/p163v/histopathology/TCGA/ffpe/299/'




slide_meta = pd.read_csv("../metadata/tcga_labeled_data.csv")




(slide_meta.project == "LGG").sum()




slide_meta[(slide_meta.project=="LGG") | (slide_meta.project=="GBM")]




slide_meta = pd.read_csv("../metadata/tcga_labeled_data.csv")

# 50 random slides

# myx = np.random.choice(np.array(range(len(slide_meta[slide_meta.project=="GBM"].slide_id))), 50)
myx = range(len(slide_meta[(slide_meta.project=="LGG") | (slide_meta.project=="GBM")].slide_id))
gbm_slides = slide_meta[(slide_meta.project=="LGG") | (slide_meta.project=="GBM")].slide_id.iloc[myx]

test_labels = slide_meta[(slide_meta.project=="LGG") | (slide_meta.project=="GBM")].labels.iloc[myx]

#all_files = [x for x in slide_annots.file if os.path.isfile(path_to_extracted_features + "/" + x)]
#    all_features = {file: h5py.File(path_to_extracted_features + "/" + file, 'r')['feats'][:] for file in all_files}




test_labels.mean()




test_labels.__len__()




test_features = [h5py.File(path_to_extracted_features + "/" + file + ".h5", 'r')['feats'][:] for file in gbm_slides]


# # Loss

# We use the loss as the early stopping criteria
# 



model_list = list()
attention_list = list()
prob_list = list()
pred_list = list()
cv =  lambda x: np.std(x) / np.mean(x)




test_data = RetCCLFeatureLoaderMem(test_features, np.array(test_labels), patches_per_iter='all')

RetCCLTest = DataLoader(test_data, batch_size=1, num_workers=1)#, sampler=valid_Sampler)




for ii in range(len(runs)):
    
    arts = runs[ii].logged_artifacts()
    arts_dict = {a.name.removesuffix(':'+a.version).split('-')[0]: a for a in arts}
    checkpoint_folder_name = arts_dict['model'].name.split('-')[1].removesuffix(':'+arts_dict['model'].version)
    args = runs[0].config

    model = Attention(2048, lr=args['lr'], weight_decay=args['weight_decay'], hidden_dim=args['hidden_dim'], attention_dim=args['attention_dim'], class_weights=torch.tensor(float(args['class_weights'])))
    chkpt_file = glob.glob('lightning_logs/'+checkpoint_folder_name+'/checkpoints/best_loss*')[0]
    model = model.load_from_checkpoint(chkpt_file, map_location=torch.device('cpu'))
    model.eval()
    model_list.append(model)
    model_forward = [model.forward(x.to(model.device)) for x,y in iter(RetCCLTest)]
    model_attention = [x[2].detach().numpy() for x in model_forward]
    model_prob = [x[0].detach().numpy() for x in model_forward]
    model_pred = [x[1].detach().numpy() for x in model_forward]

    attention_list.append(model_attention)
    prob_list.append(model_prob)
    pred_list.append(model_pred)
    




prob_list = [np.concatenate(x) for x in prob_list]
pred_list = [np.concatenate(x) for x in pred_list]




prob_test = np.mean(np.vstack(prob_list), axis=0)




prob_test = np.mean(np.vstack([sigmoid_array(x) for x in prob_list]), axis=0)







fig, ax = plt.subplots()
RocCurveDisplay.from_predictions(
    np.array(test_labels),
    prob_test,
    name="CT Prediction Images",
    color="darkorange",
    plot_chance_level=True,
    ax=ax
)
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("")
plt.legend()
plt.savefig("/home/p163v/histopathology/Notebooks/GBM_Validation_ROC.png")
plt.show()




fig, ax = plt.subplots()
PrecisionRecallDisplay.from_predictions(
    np.array(test_labels),
    prob_test,
    name="CT Prediction Images",
    color="darkorange",
    plot_chance_level=True,
    ax=ax
)
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("")
plt.legend()
plt.savefig("/home/p163v/histopathology/Notebooks/GBM_Validation_AUPR.png")
plt.show()

