#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os


import h5py
import wandb
import zarr

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
from src.model.SimpleMILModels import Attention, MaxMIL, AttentionResNet, TransformerMIL
from src.transformer_from_scratch.model.position_embeddings import TrainedPositionalEmbedding2D
from src.dataloaders.DataLoaders import RetCCLFeatureLoader, RetCCLFeatureLoaderMem, RetCCLFeatureLoaderContiguousPatches
from src.utils.embedding_loaders import h5py_loader, pt_loader, zarr_loader, load_features
from src.utils.eval_utils import compute_confusion_matrix
from src.utils.parser import get_args


def main(args):

    ## Create a group tag for saving the models from the CV folds into wandb 
    random_group_tag = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))


    slide_meta = pd.read_csv(args.slide_annot_path)
    ct_scoring = pd.read_csv(args.label_path)

    slide_meta_wbt = wandb.Table(dataframe=slide_meta)
    ct_scoring_wbt = wandb.Table(dataframe=ct_scoring)

    labels_artifact = wandb.Artifact("dataset_labels", type="labels")
    labels_artifact.add(slide_meta_wbt, "slide_meta")
    labels_artifact.add(ct_scoring_wbt, "ct_scoring")

    ct_scoring["txt_idat"] = ct_scoring["idat"].astype("str")
    ct_scoring.index = ct_scoring.txt_idat
    slide_meta.index = slide_meta.idat
    ct_scoring = ct_scoring.drop("txt_idat", axis=1)
    slide_meta = slide_meta.drop("idat", axis=1)
    slide_annots = slide_meta.join(ct_scoring, lsuffix="l")


    slide_annots['file'] = slide_annots.uuid

    slide_annots.index = slide_annots.uuid

    # oncotree_map = pd.read_csv("../metadata/MappingClassifierToOncotree.csv")

    # slide_annots = slide_annots.merge(oncotree_map, left_on="max_super_family_class", right_on="Super Family", how="left")

    # if args.oncotree_filter is not None:
    #     slide_annots = slide_annots.loc[slide_annots['Oncotree code'].isin(args.oncotree_filter)]
    #     slide_annots = slide_annots.loc[slide_annots.max_cal_v11 * 100 >= args.classifier_confidence]


    ## Load in precomputed splits for the datasets - for comparison sake between different models they are fixed. 
    splits = os.listdir(args.cv_split_path)
    splits.sort()

    all_slides = [sum([pd.read_csv(args.cv_split_path + split +"/" + file).slide.tolist() for file in os.listdir(args.cv_split_path + split + "/")],[]) for split in splits]
    all_slides = np.unique(sum(all_slides, []))
    slide_annots = slide_annots.loc[all_slides]

    if args.metadata_column is not None:
        print("Using metadata column: ", args.metadata_column)
        extra_features = F.one_hot(torch.LongTensor(slide_annots.loc[:,args.metadata_column].factorize()[0]))
        extra_features = extra_features.numpy().astype(np.float16)
        all_features = load_features(args.embedding, args.patch_size, slide_annots, extra_features)
    else:
        all_features = load_features(args.embedding, args.patch_size, slide_annots)

    # Sanity check
    print(len(all_features))

    ## Run over each split
    for splt in splits:
        wandb_run = wandb.init(project="UKHD_RetCLL_299_CT", 
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


        train_slide = train_set.loc[train_set["patches"]>=args.min_patches_per_pat].slide.tolist()
        valid_slide = valid_set.loc[valid_set["patches"]>=args.min_patches_per_pat].slide.tolist()
        test_slide = test_set.loc[test_set["patches"]>=args.min_patches_per_pat].slide.tolist()

        train_files = [x for x in train_slide] # here in case file path needs to be added later
        valid_files = [x for x in valid_slide]
        test_files = [x for x in test_slide]

        train_file_list = [all_features[x] for x in train_files]
        valid_file_list = [all_features[x] for x in valid_files]
        test_file_list = [all_features[x] for x in test_files]

        train_labels = np.abs(1-slide_annots.loc[train_slide].CT_class.factorize(sort=True)[0])
        valid_labels = np.abs(1-slide_annots.loc[valid_slide].CT_class.factorize(sort=True)[0])
        test_labels = np.abs(1-slide_annots.loc[test_slide].CT_class.factorize(sort=True)[0])

        ## Do we use a random subset of tiles for each patient to train, or do we train 1 patient at a time?
        if args.training_strategy=="random_tiles":
            train_data = RetCCLFeatureLoaderMem(train_file_list, train_labels, patches_per_iter=args.patches_per_pat)
            valid_data = RetCCLFeatureLoaderMem(valid_file_list, valid_labels, patches_per_iter=args.patches_per_pat)
            # test_data = RetCCLFeatureLoaderMem(test_file_list, test_labels, patches_per_iter='all')
        elif args.training_strategy=="all_tiles":
            train_data = RetCCLFeatureLoaderMem(train_file_list, train_labels, patches_per_iter='all')
            valid_data = RetCCLFeatureLoaderMem(valid_file_list, valid_labels, patches_per_iter='all')
            # test_data = RetCCLFeatureLoaderMem(test_file_list, test_labels, patches_per_iter='all')
        elif args.training_strategy=="single_superpatch":
            if args.position_aware_transformer:
                train_dataloader = RetCCLFeatureLoaderContiguousPatches(train_file_list, train_labels, square_size = args.superpatch_size, return_coords=True)
                valid_dataloader = RetCCLFeatureLoaderContiguousPatches(valid_file_list, valid_labels, square_size = args.superpatch_size, return_coords=True)
            else:
                train_dataloader = RetCCLFeatureLoaderContiguousPatches(train_file_list, train_labels, square_size = args.superpatch_size, return_coords=False)
                valid_dataloader = RetCCLFeatureLoaderContiguousPatches(valid_file_list, valid_labels, square_size = args.superpatch_size, return_coords=False)

        # Obsessively save everything to wandb
        fold_artifact = wandb.Artifact("fold", type="cv_fold")
        train_slide_wbt = wandb.Table(dataframe=pd.DataFrame(train_slide))
        valid_slide_wbt = wandb.Table(dataframe=pd.DataFrame(valid_slide))
        test_slide_wbt = wandb.Table(dataframe=pd.DataFrame(test_slide))
        fold_artifact.add(train_slide_wbt, "training")
        fold_artifact.add(valid_slide_wbt, "validation")
        fold_artifact.add(test_slide_wbt, "testing")
        wandb_logger.experiment.log_artifact(fold_artifact)

        counts = np.bincount(train_labels)

        ## CT is a rarer phenomena, so we need to over-emphazise the CT+ class 
        pos_weight = counts[0]/counts[1]

        valid_counts = np.bincount(valid_labels)

        ## 
        # if args.training_strategy=="random_tiles":
        if args.training_strategy=="all_tiles":
            train_dataloader = DataLoader(train_data, batch_size=1, num_workers=9, shuffle=True)
            valid_dataloader = DataLoader(valid_data, batch_size=1, num_workers=4, shuffle=True)
        else:
            train_dataloader = DataLoader(train_data, batch_size=args.batch_size, num_workers=9)#, sampler=Sampler)
            valid_dataloader = DataLoader(valid_data, batch_size=args.batch_size, num_workers=4)#, sampler=valid_Sampler)

        
        if args.model=="Attention":
            model = Attention(test_file_list[0].shape[1], lr=args.lr, weight_decay=args.weight_decay, hidden_dim=args.hidden_dim, attention_dim=args.attention_dim, class_weights=torch.Tensor([pos_weight]))
        elif args.model=="Max":
            model = MaxMIL(test_file_list[0].shape[1], lr=args.lr, weight_decay=args.weight_decay, hidden_dim=args.hidden_dim, class_weights=torch.Tensor([pos_weight]))
        elif args.model=="AttentionResNet":
            model = AttentionResNet(test_file_list[0].shape[1], lr=args.lr, weight_decay=args.weight_decay, hidden_dim=args.hidden_dim, attention_dim=args.attention_dim, class_weights=torch.Tensor([pos_weight]))
        elif args.model=="TransformerMIL":
            assert args.n_heads is not None
            if args.position_aware_transformer:
                assert args.superpatch_size is not None
                pos_enc = TrainedPositionalEmbedding2D(args.superpatch_size, args.superpatch_size, args.hidden_dim[0], cls_token = args.embed_extra_tokens)
                model = TransformerMIL(test_file_list[0].shape[1], lr=args.lr, weight_decay=args.weight_decay, hidden_dim=args.hidden_dim, n_heads=args.n_heads, class_weights=torch.Tensor([pos_weight]), position_encoding=pos_enc)
            else: 
                model = TransformerMIL(test_file_list[0].shape[1], lr=args.lr, weight_decay=args.weight_decay, hidden_dim=args.hidden_dim, n_heads=args.n_heads, class_weights=torch.Tensor([pos_weight]))
        
        ## We checkpoint on 3 different metrics, the loss, the error and the F1 score. 
        
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

        ## If we go 1 patient at a time, we gradient accumulate 
        if args.training_strategy=="all_tiles":
            trainer = L.Trainer(max_epochs=args.max_epochs, log_every_n_steps=1, logger=wandb_logger, callbacks=[checkpoint_error,checkpoint_loss, checkpoint_f1], accumulate_grad_batches=args.batch_size, gradient_clip_val = args.clip_grad) # limit_train_batches=100,
        else: 
            trainer = L.Trainer(max_epochs=args.max_epochs, log_every_n_steps=1, logger=wandb_logger, callbacks=[checkpoint_error,checkpoint_loss, checkpoint_f1], gradient_clip_val = args.clip_grad) # limit_train_batches=100,
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

        
        ## Finish this run so we can record the next fold at the top of the loop
        wandb.finish()


if __name__ == '__main__':

    args = get_args()
    # Parse the user inputs and defaults (returns a argparse.Namespace)
   

    ## Needed on some servers at DKFZ 
    os.environ['HTTP_PROXY']="http://www-int.dkfz-heidelberg.de:80"
    os.environ['HTTPS_PROXY']="http://www-int.dkfz-heidelberg.de:80"

    main(args)