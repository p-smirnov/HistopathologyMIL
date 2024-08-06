#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
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
from src.model.SimpleMILModels import Attention, MaxMIL, AttentionResNet
from src.dataloaders.DataLoaders import RetCCLFeatureLoader, RetCCLFeatureLoaderMem


from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--model", type=str, default="Attention")


# Trainer arguments
parser.add_argument("--hidden_dim", type=int, default=[512], nargs="+")
parser.add_argument("--attention_dim", type=int, default=[256], nargs="+")

# Hyperparameters for the model
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--weight_decay", type=float, default=10e-4)
parser.add_argument("--max_epochs", type=int, default=100)
parser.add_argument("--patch_size", type=int, default=299)
parser.add_argument("--patches_per_pat", type=int, default=10)
parser.add_argument("--training_strategy", type=str, default='random_tiles')
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--cv_split_path", type=str, default='/home/p163v/histopathology/splits/TCGA/18032024/')

# Parse the user inputs and defaults (returns a argparse.Namespace)
args = parser.parse_args()

if args.training_strategy=="all_tiles":
    args.patches_per_pat = int(10)


## Set up W&B logger

random_group_tag = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))

# wandb_logger = WandbLogger(project="UKHD_RetCLL_299_CT", 
#                            group=random_group_tag,
#                            log_model=True, 
#                            offline=False, 
#                            settings=wandb.Settings(start_method="fork"),
#                            tags=['Dataset2', 'CV'])

# wandb_logger.log_hyperparams(args)


# from AttentionMIL_model import Attention

path_to_extracted_features = '/home/p163v/histopathology/TCGA/ffpe/' + str(args.patch_size) + '/'



import os
os.environ['HTTP_PROXY']="http://www-int.dkfz-heidelberg.de:80"
os.environ['HTTPS_PROXY']="http://www-int.dkfz-heidelberg.de:80"

# os.environ['TENSORBOARD_BINARY'] = '/home/p163v/mambaforge/envs/marugoto/bin/tensorboard'


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



slide_meta = pd.read_csv("/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/metadata/tcga_labeled_data.csv")

slide_meta_wbt = wandb.Table(dataframe=slide_meta)

labels_artifact = wandb.Artifact("dataset_labels", type="labels")
labels_artifact.add(slide_meta_wbt, "slide_meta")



slide_meta['file'] = slide_meta.slide_id + ".h5"
slide_meta.set_index('slide_id', inplace=True)

splits = os.listdir(args.cv_split_path)
splits.sort()


all_files = [x for x in slide_meta.file if os.path.isfile(path_to_extracted_features + "/" + x)]

all_features = {file: h5py.File(path_to_extracted_features + "/" + file, 'r')['feats'][:] for file in all_files}

for splt in splits:

    wandb_run = wandb.init(project="TCGA_RetCLL_299_CT", 
                           group=random_group_tag,
                        #    log_model=True, 
                        #    offline=False, 
                        #    settings=wandb.Settings(start_method="fork"),
                           tags=['Dataset2', 'CV', 'fold_'+splt, 'f1'])

    wandb_logger = WandbLogger(log_model=True)

    wandb_logger.log_hyperparams(args)
    wandb_logger.experiment.log_artifact(labels_artifact)

    split_path = args.cv_split_path + splt + "/"

    train_set = pd.read_csv(split_path + "train_set.csv")
    valid_set = pd.read_csv(split_path + "valid_set.csv")
    test_set = pd.read_csv(split_path + "test_set.csv")


    # train_files = train_set.loc[train_set["patches"]>=args.patches_per_pat].features.tolist()
    # valid_files = valid_set.loc[valid_set["patches"]>=args.patches_per_pat].features.tolist()
    # test_files = test_set.loc[test_set["patches"]>=args.patches_per_pat].features.tolist()

    train_slide = train_set.loc[train_set["patches"]>=args.patches_per_pat].slide_id.tolist()
    valid_slide = valid_set.loc[valid_set["patches"]>=args.patches_per_pat].slide_id.tolist()
    test_slide = test_set.loc[test_set["patches"]>=args.patches_per_pat].slide_id.tolist()

    train_files = [x + '.h5' for x in train_slide]
    valid_files = [x + '.h5' for x in valid_slide]
    test_files = [x + '.h5' for x in test_slide]

    train_file_list = [all_features[x] for x in train_files]
    valid_file_list = [all_features[x] for x in valid_files]
    test_file_list = [all_features[x] for x in test_files]

    train_labels = np.abs(1-slide_meta.loc[train_slide].CT_Status.factorize(sort=True)[0])
    valid_labels = np.abs(1-slide_meta.loc[valid_slide].CT_Status.factorize(sort=True)[0])
    test_labels = np.abs(1-slide_meta.loc[test_slide].CT_Status.factorize(sort=True)[0])

    if args.training_strategy=="random_tiles":
        train_data = RetCCLFeatureLoaderMem(train_file_list, train_labels, patches_per_iter=args.patches_per_pat)
        valid_data = RetCCLFeatureLoaderMem(valid_file_list, valid_labels, patches_per_iter=args.patches_per_pat)
        test_data = RetCCLFeatureLoaderMem(test_file_list, test_labels, patches_per_iter='all')
    elif args.training_strategy=="all_tiles":
        train_data = RetCCLFeatureLoaderMem(train_file_list, train_labels, patches_per_iter='all')
        valid_data = RetCCLFeatureLoaderMem(valid_file_list, valid_labels, patches_per_iter='all')
        test_data = RetCCLFeatureLoaderMem(test_file_list, test_labels, patches_per_iter='all')


    fold_artifact = wandb.Artifact("fold", type="cv_fold")
    train_slide_wbt = wandb.Table(dataframe=pd.DataFrame(train_slide))
    valid_slide_wbt = wandb.Table(dataframe=pd.DataFrame(valid_slide))
    test_slide_wbt = wandb.Table(dataframe=pd.DataFrame(test_slide))
    fold_artifact.add(train_slide_wbt, "training")
    fold_artifact.add(valid_slide_wbt, "validation")
    fold_artifact.add(test_slide_wbt, "testing")
    wandb_logger.experiment.log_artifact(fold_artifact)

    counts = np.bincount(train_labels)

    pos_weight = counts[0]/counts[1]

    valid_counts = np.bincount(valid_labels)

    if args.training_strategy=="random_tiles":
        train_dataloader = DataLoader(train_data, batch_size=64, num_workers=9)#, sampler=Sampler)
        valid_dataloader = DataLoader(valid_data, batch_size=128, num_workers=4)#, sampler=valid_Sampler)
    elif args.training_strategy=="all_tiles":
        train_dataloader = DataLoader(train_data, batch_size=1, num_workers=9, shuffle=True)
        valid_dataloader = DataLoader(valid_data, batch_size=1, num_workers=4, shuffle=True)


    if args.model=="Attention":
        model = Attention(2048, lr=args.lr, weight_decay=args.weight_decay, hidden_dim=args.hidden_dim, attention_dim=args.attention_dim, class_weights=torch.Tensor([pos_weight]))
    elif args.model=="Max":
        model = MaxMIL(2048, lr=args.lr, weight_decay=args.weight_decay, hidden_dim=args.hidden_dim, class_weights=torch.Tensor([pos_weight]))
    elif args.model=="AttentionResNet":
        model = AttentionResNet(2048, lr=args.lr, weight_decay=args.weight_decay, hidden_dim=args.hidden_dim, attention_dim=args.attention_dim, class_weights=torch.Tensor([pos_weight]))

    checkpoint_error = ModelCheckpoint(save_top_k=1, 
                                          monitor="valid_error",
                                          mode="min",
                                          save_last=True,
                                          filename='best_error_{epoch}-{valid_error:.2f}')

    checkpoint_f1 = ModelCheckpoint(save_top_k=1, 
                                          monitor="valid_f1",
                                          mode="max",
                                          save_last=True,
                                          filename='best_f1_{epoch}-{valid_f1:.2f}')

    checkpoint_loss = ModelCheckpoint(save_top_k=1, 
                                          monitor="valid_loss",
                                          mode="min",
                                          save_last=True,
                                          filename='best_loss_{epoch}-{valid_loss:.2f}')

    trainer = L.Trainer(max_epochs=args.max_epochs, log_every_n_steps=1, logger=wandb_logger, callbacks=[checkpoint_error,checkpoint_loss, checkpoint_f1]) # limit_train_batches=100,
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

    print(checkpoint_f1.best_model_path)

    model = model.load_from_checkpoint(checkpoint_error.best_model_path)
    model.eval()

    train_data = RetCCLFeatureLoaderMem(train_file_list,train_labels, patches_per_iter='all')
    train_dataloader = DataLoader(train_data, batch_size=1, num_workers=4, shuffle=False)

    valid_data = RetCCLFeatureLoaderMem(valid_file_list,valid_labels, patches_per_iter='all')
    valid_dataloader = DataLoader(valid_data, batch_size=1, num_workers=4, shuffle=False)

    test_dataloader = DataLoader(test_data, batch_size=1, num_workers=4, shuffle=False)

    # TODO: determine if this leaves some sort of pointer to GPU memory which would stay around through the next fold
    # This seems to explode the GPU memory usage, maybe if I immediately discard the attn maps it will be better?
    # train_inference = [model(torch.tensor(x).to(model.device)) for x,y in iter(train_dataloader)]
    # train_preds = [x[1].detach().item() for x in train_inference]
    # train_probs = [x[0].detach().item() for x in train_inference]

    train_preds = [model(torch.tensor(x).to(model.device))[1].detach().item() for x,y in iter(train_dataloader)]
    train_probs = [model(torch.tensor(x).to(model.device))[0].detach().item() for x,y in iter(train_dataloader)]   
    train_confusion = compute_confusion_matrix(train_labels.astype(int), np.array(train_preds).astype(int))


    # valid_inference = [model(torch.tensor(x).to(model.device)) for x,y in iter(valid_dataloader)]
    # valid_preds = [x[1].detach().item() for x in valid_inference]
    # valid_probs = [x[0].detach().item() for x in valid_inference]
    valid_preds = [model(torch.tensor(x).to(model.device))[1].detach().item() for x,y in iter(valid_dataloader)]
    valid_probs = [model(torch.tensor(x).to(model.device))[0].detach().item() for x,y in iter(valid_dataloader)]
    valid_confusion = compute_confusion_matrix(valid_labels.astype(int), np.array(valid_preds).astype(int))

    # trainer.test(model, dataloaders=test_dataloader) 

    # test_inference = [model(torch.tensor(x).to(model.device)) for x,y in iter(test_dataloader)]
    # test_preds = [x[1].detach().item() for x in test_inference]
    # test_probs = [x[0].detach().item() for x in test_inference]
    test_preds = [model(torch.tensor(x).to(model.device))[1].detach().item() for x,y in iter(test_dataloader)]
    test_probs = [model(torch.tensor(x).to(model.device))[0].detach().item() for x,y in iter(test_dataloader)]
    test_confusion = compute_confusion_matrix(test_labels.astype(int), np.array(test_preds).astype(int))

    pred_artifact = wandb.Artifact("preds_error", type="predictions_error")
    pred_artifact.add(wandb.Table(dataframe=pd.DataFrame(train_preds)), "training")
    pred_artifact.add(wandb.Table(dataframe=pd.DataFrame(valid_preds)), "validation")
    pred_artifact.add(wandb.Table(dataframe=pd.DataFrame(test_preds)), "testing")
    pred_artifact.add(wandb.Table(dataframe=pd.DataFrame(train_probs)), "training_probs")
    pred_artifact.add(wandb.Table(dataframe=pd.DataFrame(valid_probs)), "validation_probs")
    pred_artifact.add(wandb.Table(dataframe=pd.DataFrame(test_probs)), "testing_probs")
    wandb_logger.experiment.log_artifact(pred_artifact)

    conf_artifact = wandb.Artifact("confusion_error", type="predictions_error")
    conf_artifact.add(wandb.Table(dataframe=pd.DataFrame(train_confusion)), "training")
    conf_artifact.add(wandb.Table(dataframe=pd.DataFrame(valid_confusion)), "validation")
    conf_artifact.add(wandb.Table(dataframe=pd.DataFrame(test_confusion)), "testing")
    wandb_logger.experiment.log_artifact(conf_artifact)

    model = model.load_from_checkpoint(checkpoint_f1.best_model_path)
    model.eval()

    train_data = RetCCLFeatureLoaderMem(train_file_list,train_labels, patches_per_iter='all')
    train_dataloader = DataLoader(train_data, batch_size=1, num_workers=4, shuffle=False)

    valid_data = RetCCLFeatureLoaderMem(valid_file_list,valid_labels, patches_per_iter='all')
    valid_dataloader = DataLoader(valid_data, batch_size=1, num_workers=4, shuffle=False)

    test_dataloader = DataLoader(test_data, batch_size=1, num_workers=4, shuffle=False)

    # TODO: determine if this leaves some sort of pointer to GPU memory which would stay around through the next fold
    # This seems to explode the GPU memory usage, maybe if I immediately discard the attn maps it will be better?
    # train_inference = [model(torch.tensor(x).to(model.device)) for x,y in iter(train_dataloader)]
    # train_preds = [x[1].detach().item() for x in train_inference]
    # train_probs = [x[0].detach().item() for x in train_inference]

    train_preds = [model(torch.tensor(x).to(model.device))[1].detach().item() for x,y in iter(train_dataloader)]
    train_probs = [model(torch.tensor(x).to(model.device))[0].detach().item() for x,y in iter(train_dataloader)]   
    train_confusion = compute_confusion_matrix(train_labels.astype(int), np.array(train_preds).astype(int))


    # valid_inference = [model(torch.tensor(x).to(model.device)) for x,y in iter(valid_dataloader)]
    # valid_preds = [x[1].detach().item() for x in valid_inference]
    # valid_probs = [x[0].detach().item() for x in valid_inference]
    valid_preds = [model(torch.tensor(x).to(model.device))[1].detach().item() for x,y in iter(valid_dataloader)]
    valid_probs = [model(torch.tensor(x).to(model.device))[0].detach().item() for x,y in iter(valid_dataloader)]
    valid_confusion = compute_confusion_matrix(valid_labels.astype(int), np.array(valid_preds).astype(int))

    # trainer.test(model, dataloaders=test_dataloader) 

    # test_inference = [model(torch.tensor(x).to(model.device)) for x,y in iter(test_dataloader)]
    # test_preds = [x[1].detach().item() for x in test_inference]
    # test_probs = [x[0].detach().item() for x in test_inference]
    test_preds = [model(torch.tensor(x).to(model.device))[1].detach().item() for x,y in iter(test_dataloader)]
    test_probs = [model(torch.tensor(x).to(model.device))[0].detach().item() for x,y in iter(test_dataloader)]
    test_confusion = compute_confusion_matrix(test_labels.astype(int), np.array(test_preds).astype(int))

    pred_artifact = wandb.Artifact("preds_f1", type="predictions_f1")
    pred_artifact.add(wandb.Table(dataframe=pd.DataFrame(train_preds)), "training")
    pred_artifact.add(wandb.Table(dataframe=pd.DataFrame(valid_preds)), "validation")
    pred_artifact.add(wandb.Table(dataframe=pd.DataFrame(test_preds)), "testing")
    pred_artifact.add(wandb.Table(dataframe=pd.DataFrame(train_probs)), "training_probs")
    pred_artifact.add(wandb.Table(dataframe=pd.DataFrame(valid_probs)), "validation_probs")
    pred_artifact.add(wandb.Table(dataframe=pd.DataFrame(test_probs)), "testing_probs")
    wandb_logger.experiment.log_artifact(pred_artifact)

    conf_artifact = wandb.Artifact("confusion_f1", type="predictions_f1")
    conf_artifact.add(wandb.Table(dataframe=pd.DataFrame(train_confusion)), "training")
    conf_artifact.add(wandb.Table(dataframe=pd.DataFrame(valid_confusion)), "validation")
    conf_artifact.add(wandb.Table(dataframe=pd.DataFrame(test_confusion)), "testing")
    wandb_logger.experiment.log_artifact(conf_artifact)


    model = model.load_from_checkpoint(checkpoint_loss.best_model_path)
    model.eval()

    train_data = RetCCLFeatureLoaderMem(train_file_list,train_labels, patches_per_iter='all')
    train_dataloader = DataLoader(train_data, batch_size=1, num_workers=4, shuffle=False)

    valid_data = RetCCLFeatureLoaderMem(valid_file_list,valid_labels, patches_per_iter='all')
    valid_dataloader = DataLoader(valid_data, batch_size=1, num_workers=4, shuffle=False)

    test_dataloader = DataLoader(test_data, batch_size=1, num_workers=4, shuffle=False)

    # TODO: determine if this leaves some sort of pointer to GPU memory which would stay around through the next fold
    # This seems to explode the GPU memory usage, maybe if I immediately discard the attn maps it will be better?
    # train_inference = [model(torch.tensor(x).to(model.device)) for x,y in iter(train_dataloader)]
    # train_preds = [x[1].detach().item() for x in train_inference]
    # train_probs = [x[0].detach().item() for x in train_inference]

    train_preds = [model(torch.tensor(x).to(model.device))[1].detach().item() for x,y in iter(train_dataloader)]
    train_probs = [model(torch.tensor(x).to(model.device))[0].detach().item() for x,y in iter(train_dataloader)]   
    train_confusion = compute_confusion_matrix(train_labels.astype(int), np.array(train_preds).astype(int))


    # valid_inference = [model(torch.tensor(x).to(model.device)) for x,y in iter(valid_dataloader)]
    # valid_preds = [x[1].detach().item() for x in valid_inference]
    # valid_probs = [x[0].detach().item() for x in valid_inference]
    valid_preds = [model(torch.tensor(x).to(model.device))[1].detach().item() for x,y in iter(valid_dataloader)]
    valid_probs = [model(torch.tensor(x).to(model.device))[0].detach().item() for x,y in iter(valid_dataloader)]
    valid_confusion = compute_confusion_matrix(valid_labels.astype(int), np.array(valid_preds).astype(int))

    # trainer.test(model, dataloaders=test_dataloader) 

    # test_inference = [model(torch.tensor(x).to(model.device)) for x,y in iter(test_dataloader)]
    # test_preds = [x[1].detach().item() for x in test_inference]
    # test_probs = [x[0].detach().item() for x in test_inference]
    test_preds = [model(torch.tensor(x).to(model.device))[1].detach().item() for x,y in iter(test_dataloader)]
    test_probs = [model(torch.tensor(x).to(model.device))[0].detach().item() for x,y in iter(test_dataloader)]
    test_confusion = compute_confusion_matrix(test_labels.astype(int), np.array(test_preds).astype(int))

    pred_artifact = wandb.Artifact("preds_loss", type="predictions_loss")
    pred_artifact.add(wandb.Table(dataframe=pd.DataFrame(train_preds)), "training")
    pred_artifact.add(wandb.Table(dataframe=pd.DataFrame(valid_preds)), "validation")
    pred_artifact.add(wandb.Table(dataframe=pd.DataFrame(test_preds)), "testing")
    pred_artifact.add(wandb.Table(dataframe=pd.DataFrame(train_probs)), "training_probs")
    pred_artifact.add(wandb.Table(dataframe=pd.DataFrame(valid_probs)), "validation_probs")
    pred_artifact.add(wandb.Table(dataframe=pd.DataFrame(test_probs)), "testing_probs")
    wandb_logger.experiment.log_artifact(pred_artifact)

    conf_artifact = wandb.Artifact("confusion_loss", type="predictions_loss")
    conf_artifact.add(wandb.Table(dataframe=pd.DataFrame(train_confusion)), "training")
    conf_artifact.add(wandb.Table(dataframe=pd.DataFrame(valid_confusion)), "validation")
    conf_artifact.add(wandb.Table(dataframe=pd.DataFrame(test_confusion)), "testing")
    wandb_logger.experiment.log_artifact(conf_artifact)
    wandb.finish()


