#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import wandb
import json
import glob
from tqdm import tqdm

import string
import random

import torch 
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from src.model.SimpleMILModels import Attention, MaxMIL, AttentionResNet, TransformerMIL
from src.dataloaders.DataLoaders import FromDiskFeatureLoader
from src.utils.embedding_loaders import load_features

from argparse import ArgumentParser


def main(runs, args):
    """
    Main function for running inference on all embedded slides, regardless of whether they are labelled, and saving the results to disk for each model.

    Parameters:
    runs (list): A list of runs to perform inference on.
    args (object): An object containing the arguments for the function.

    Returns:
    None
    """

    # In this file, unlike in other training scripts, we run over all the embedded slides, regardless of whether they are labelled, and save the results to disk for each model.
    
    # First, we check that all models were trained on the same datasets:

    runs_configs = [json.loads(x.json_config) for x in runs]
    if not all([x[y] == runs_configs[0][y] for x in runs_configs] for y in ['embedding', 'cv_split_path', 'label_path']):
        print("Something is different between the datasets used to train the runs, aborting")
        raise RuntimeError

    num_runs = len(runs)
    # Since we checked that all runs use the same data, we can safely load in all the files now, only once, and create a matrix to store results

    parse_json_to_args(runs_configs[0], args)
    print(args)

    slide_annots = pd.read_csv(args.slide_annot_path)
    # ct_scoring = pd.read_csv(args.label_path)
    slide_annots['file'] = slide_annots['uuid']

    all_slide_data = FromDiskFeatureLoader(args.embedding, slide_annots['uuid'], args.patch_size)

    train_dataloader = DataLoader(all_slide_data, batch_size=1, num_workers=12, shuffle=False)

    output_preds = np.zeros((len(all_slide_data), num_runs))
    output_probs = np.zeros((len(all_slide_data), num_runs))

    model_list = list()

    for i, run in enumerate(runs):
        arts = run.logged_artifacts()
        arts_dict = {a.name.removesuffix(':'+a.version).split('-')[0]: a for a in arts}
        checkpoint_folder_name = arts_dict['model'].name.split('-')[1].removesuffix(':'+arts_dict['model'].version)
        parse_json_to_args(runs_configs[i], args)

        if args.model=="Attention":
            model = Attention(all_slide_data.feature_size, lr=args.lr, weight_decay=args.weight_decay, hidden_dim=args.hidden_dim, attention_dim=args.attention_dim, class_weights=torch.Tensor([1]))
        elif args.model=="Max":
            model = MaxMIL(all_slide_data.feature_size, lr=args.lr, weight_decay=args.weight_decay, hidden_dim=args.hidden_dim, class_weights=torch.Tensor([1]))
        elif args.model=="AttentionResNet":
            model = AttentionResNet(all_slide_data.feature_size, lr=args.lr, weight_decay=args.weight_decay, hidden_dim=args.hidden_dim, attention_dim=args.attention_dim, class_weights=torch.Tensor([1]))
        elif args.model=="TransformerMIL":
            assert args.n_heads is not None
            model = TransformerMIL(all_slide_data.feature_size, lr=args.lr, weight_decay=args.weight_decay, hidden_dim=args.hidden_dim, n_heads=args.n_heads, class_weights=torch.Tensor([1]))

        chkpt_file = glob.glob('lightning_logs/'+checkpoint_folder_name+'/checkpoints/best_loss*')[0]
        model = model.load_from_checkpoint(chkpt_file, map_location=torch.device('cuda'))
        model.eval()
        model_list.append(model)
    
    with torch.inference_mode():
        for j, x in enumerate(tqdm(train_dataloader)):
            for i, model in enumerate(model_list):
                output_probs[j,i], output_preds[j,i] = tuple_to_cpu(model.forward(x.to(model.device))[0:2])


    np.save(args.run_prefix + '_output_preds.npy', output_preds)
    np.save(args.run_prefix + '_output_probs.npy', output_probs)
    np.save(args.run_prefix + '_slide_names.npy', np.array(all_slide_data.slides))
    print('Inference complete')

def parse_json_to_args(json_parsed, args):
    """
    This function parses a JSON object and sets the corresponding attributes in the given args Namespace object.
    It also sets defaults that may have been missing from runs prior to those arguments being added to the train script. 

    Parameters:
    json_parsed (dict): A dictionary containing the JSON data to be parsed.
    args (object): An object whose attributes will be set based on the JSON data.

    Returns:
    None
    """
    # wandb_args = json.loads(json_string)
    wandb_args = json_parsed 
    for key in wandb_args:
        cur_arg = wandb_args[key]
        if '../' in str(cur_arg['value']):
            # This fix is here because I used relative paths that got changed 
            cur_arg['value'] = '/home/p163v/histopathology/' + cur_arg['value'].lstrip('../')
        setattr(args, key, cur_arg['value'])
    if not hasattr(args, 'slide_annot_path'):
        args.slide_annot_path = "/home/p163v/histopathology/metadata/labels_with_new_batch.csv"
    if not hasattr(args, 'label_path'):
        args.label_path = "/home/p163v/histopathology/metadata/CT_3_Class_Draft.csv"
    if args.training_strategy=="all_tiles":
        args.patches_per_pat = int(10)

def tuple_to_cpu(tpl):
    """
    This function takes a tuple of PyTorch tensors and moves them to the CPU.

    Parameters:
        tpl (tuple): A tuple of PyTorch tensors.

    Returns:
        tuple: A tuple of PyTorch tensors moved to the CPU.
    """
    return tuple(x.cpu() for x in tpl)


if __name__ == '__main__':

    parser = ArgumentParser()
  
    parser.add_argument("--run_prefix", type=str, default='6X01HP')

    # All other arguments will be fetched from the argument store in WandB

    # Parse the user inputs and defaults (returns a argparse.Namespace)
    args = parser.parse_args()

    # ## Needed on some servers at DKFZ 
    # os.environ['HTTP_PROXY']="http://www-int.dkfz-heidelberg.de:80"
    # os.environ['HTTPS_PROXY']="http://www-int.dkfz-heidelberg.de:80"

    api = wandb.Api()
    runs = api.runs(path="psmirnov/UKHD_RetCLL_299_CT", filters={"group": args.run_prefix})


    main(runs, args)
