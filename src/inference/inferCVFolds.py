
import numpy as np
import pandas as pd
import os


import wandb
import glob


import torch 
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

import lightning as L
# from lightning.pytorch.loggers import WandbLogger
# from lightning.pytorch.callbacks import ModelCheckpoint


# from torch import optim, utils, Tensor
from src.model.SimpleMILModels import Attention, MaxMIL, AttentionResNet, TransformerMIL
from src.dataloaders.DataLoaders import RetCCLFeatureLoader, RetCCLFeatureLoaderMem
from src.utils.embedding_loaders import h5py_loader, pt_loader, zarr_loader, load_features
from src.utils.eval_utils import compute_confusion_matrix

from argparse import ArgumentParser

def main(args):

    api = wandb.Api()
    runs = api.runs(path="psmirnov/UKHD_RetCLL_299_CT", filters={"group": args.group_id})

    

    run_config = runs[0].config

    for key, value in run_config.items():
        setattr(args, key, value)

    slide_meta = pd.read_csv(args.slide_annot_path)
    ct_scoring = pd.read_csv(args.label_path)


    ct_scoring["txt_idat"] = ct_scoring["idat"].astype("str")
    ct_scoring.index = ct_scoring.txt_idat
    slide_meta.index = slide_meta.idat
    ct_scoring = ct_scoring.drop("txt_idat", axis=1)
    slide_meta = slide_meta.drop("idat", axis=1)
    slide_annots = slide_meta.join(ct_scoring, lsuffix="l")


    slide_annots['file'] = slide_annots.uuid

    slide_annots.index = slide_annots.uuid


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

    pos_weight = 1

    for ii, run in enumerate(runs):

        ## WandB doesn't allow modifying finalized runs, but I can create new runs referencing the original. This complicates things, but oh well...

        new_run = wandb.init(project=run.project, job_type="add_evaluation_of_CV", group=run.group + "_eval")
        new_run.log({"original_run_id": run.id})
        new_run.config.update(run.config)

        arts = run.logged_artifacts()
        arts_dict = {a.name.removesuffix(':'+a.version).split('-')[0]: a for a in arts}
        checkpoint_folder_name = arts_dict['model'].name.split('-')[1].removesuffix(':'+arts_dict['model'].version)
        train_slide = arts_dict['fold'].get('training').get_dataframe()[0]
        valid_slide = arts_dict['fold'].get('validation').get_dataframe()[0]
        test_slide = arts_dict['fold'].get('testing').get_dataframe()[0]

        train_files = [x for x in train_slide] # here in case file path needs to be added later
        valid_files = [x for x in valid_slide]
        test_files = [x for x in test_slide]

        train_file_list = [all_features[x] for x in train_files]
        valid_file_list = [all_features[x] for x in valid_files]
        test_file_list = [all_features[x] for x in test_files]

        train_labels = np.abs(1-slide_annots.loc[train_slide].CT_class.factorize(sort=True)[0])
        valid_labels = np.abs(1-slide_annots.loc[valid_slide].CT_class.factorize(sort=True)[0])
        test_labels = np.abs(1-slide_annots.loc[test_slide].CT_class.factorize(sort=True)[0])
    
        
        if args.model=="Attention":
            model = Attention(test_file_list[0].shape[1], lr=args.lr, weight_decay=args.weight_decay, hidden_dim=args.hidden_dim, attention_dim=args.attention_dim, class_weights=torch.Tensor([pos_weight]))
        elif args.model=="Max":
            model = MaxMIL(test_file_list[0].shape[1], lr=args.lr, weight_decay=args.weight_decay, hidden_dim=args.hidden_dim, class_weights=torch.Tensor([pos_weight]))
        elif args.model=="AttentionResNet":
            model = AttentionResNet(test_file_list[0].shape[1], lr=args.lr, weight_decay=args.weight_decay, hidden_dim=args.hidden_dim, attention_dim=args.attention_dim, class_weights=torch.Tensor([pos_weight]))
        elif args.model=="TransformerMIL":
            assert args.n_heads is not None
            model = TransformerMIL(test_file_list[0].shape[1], lr=args.lr, weight_decay=args.weight_decay, hidden_dim=args.hidden_dim, n_heads=args.n_heads, class_weights=torch.Tensor([pos_weight]))
        
        ########
        ## Now, we run and save the predictions for the best error model
        ########
        chkpt_error = glob.glob('lightning_logs/'+checkpoint_folder_name+'/checkpoints/best_error*')[0]
        model = model.load_from_checkpoint(chkpt_error)
        model.eval()

        train_data = RetCCLFeatureLoaderMem(train_file_list,train_labels, patches_per_iter='all')
        train_dataloader = DataLoader(train_data, batch_size=1, num_workers=4, shuffle=False)

        valid_data = RetCCLFeatureLoaderMem(valid_file_list,valid_labels, patches_per_iter='all')
        valid_dataloader = DataLoader(valid_data, batch_size=1, num_workers=4, shuffle=False)


        test_data = RetCCLFeatureLoaderMem(test_file_list, test_labels, patches_per_iter='all')
        test_dataloader = DataLoader(test_data, batch_size=1, num_workers=4, shuffle=False)

        train_preds = [model(torch.tensor(x).to(model.device))[1].detach().item() for x,y, mask in iter(train_dataloader)]
        train_probs = [model(torch.tensor(x).to(model.device))[0].detach().item() for x,y, mask in iter(train_dataloader)]   
        train_confusion = compute_confusion_matrix(train_labels.astype(int), np.array(train_preds).astype(int))


        valid_preds = [model(torch.tensor(x).to(model.device))[1].detach().item() for x,y, mask in iter(valid_dataloader)]
        valid_probs = [model(torch.tensor(x).to(model.device))[0].detach().item() for x,y, mask in iter(valid_dataloader)]
        valid_confusion = compute_confusion_matrix(valid_labels.astype(int), np.array(valid_preds).astype(int))


        test_preds = [model(torch.tensor(x).to(model.device))[1].detach().item() for x,y, mask in iter(test_dataloader)]
        test_probs = [model(torch.tensor(x).to(model.device))[0].detach().item() for x,y, mask in iter(test_dataloader)]
        test_confusion = compute_confusion_matrix(test_labels.astype(int), np.array(test_preds).astype(int))

        pred_artifact = wandb.Artifact("preds_error", type="predictions_error")
        pred_artifact.add(wandb.Table(dataframe=pd.DataFrame(train_preds)), "training")
        pred_artifact.add(wandb.Table(dataframe=pd.DataFrame(valid_preds)), "validation")
        pred_artifact.add(wandb.Table(dataframe=pd.DataFrame(test_preds)), "testing")
        pred_artifact.add(wandb.Table(dataframe=pd.DataFrame(train_probs)), "training_probs")
        pred_artifact.add(wandb.Table(dataframe=pd.DataFrame(valid_probs)), "validation_probs")
        pred_artifact.add(wandb.Table(dataframe=pd.DataFrame(test_probs)), "testing_probs")
        new_run.log_artifact(pred_artifact)

        conf_artifact = wandb.Artifact("confusion_error", type="predictions_error")
        conf_artifact.add(wandb.Table(dataframe=pd.DataFrame(train_confusion)), "training")
        conf_artifact.add(wandb.Table(dataframe=pd.DataFrame(valid_confusion)), "validation")
        conf_artifact.add(wandb.Table(dataframe=pd.DataFrame(test_confusion)), "testing")
        new_run.log_artifact(conf_artifact)

        ########
        ## Now, we run and save the predictions for the best f1 score model
        ########
        chkpt_f1 = glob.glob('lightning_logs/'+checkpoint_folder_name+'/checkpoints/best_f1*')[0]
        model = model.load_from_checkpoint(chkpt_f1)
        model.eval()

        train_data = RetCCLFeatureLoaderMem(train_file_list,train_labels, patches_per_iter='all')
        train_dataloader = DataLoader(train_data, batch_size=1, num_workers=4, shuffle=False)

        valid_data = RetCCLFeatureLoaderMem(valid_file_list,valid_labels, patches_per_iter='all')
        valid_dataloader = DataLoader(valid_data, batch_size=1, num_workers=4, shuffle=False)

        test_data = RetCCLFeatureLoaderMem(test_file_list, test_labels, patches_per_iter='all')
        test_dataloader = DataLoader(test_data, batch_size=1, num_workers=4, shuffle=False)


        train_preds = [model(torch.tensor(x).to(model.device))[1].detach().item() for x,y, mask in iter(train_dataloader)]
        train_probs = [model(torch.tensor(x).to(model.device))[0].detach().item() for x,y, mask in iter(train_dataloader)]   
        train_confusion = compute_confusion_matrix(train_labels.astype(int), np.array(train_preds).astype(int))

        valid_preds = [model(torch.tensor(x).to(model.device))[1].detach().item() for x,y, mask in iter(valid_dataloader)]
        valid_probs = [model(torch.tensor(x).to(model.device))[0].detach().item() for x,y, mask in iter(valid_dataloader)]
        valid_confusion = compute_confusion_matrix(valid_labels.astype(int), np.array(valid_preds).astype(int))

        test_preds = [model(torch.tensor(x).to(model.device))[1].detach().item() for x,y, mask in iter(test_dataloader)]
        test_probs = [model(torch.tensor(x).to(model.device))[0].detach().item() for x,y, mask in iter(test_dataloader)]
        test_confusion = compute_confusion_matrix(test_labels.astype(int), np.array(test_preds).astype(int))

        pred_artifact = wandb.Artifact("preds_f1", type="predictions_f1")
        pred_artifact.add(wandb.Table(dataframe=pd.DataFrame(train_preds)), "training")
        pred_artifact.add(wandb.Table(dataframe=pd.DataFrame(valid_preds)), "validation")
        pred_artifact.add(wandb.Table(dataframe=pd.DataFrame(test_preds)), "testing")
        pred_artifact.add(wandb.Table(dataframe=pd.DataFrame(train_probs)), "training_probs")
        pred_artifact.add(wandb.Table(dataframe=pd.DataFrame(valid_probs)), "validation_probs")
        pred_artifact.add(wandb.Table(dataframe=pd.DataFrame(test_probs)), "testing_probs")
        new_run.log_artifact(pred_artifact)

        conf_artifact = wandb.Artifact("confusion_f1", type="predictions_f1")
        conf_artifact.add(wandb.Table(dataframe=pd.DataFrame(train_confusion)), "training")
        conf_artifact.add(wandb.Table(dataframe=pd.DataFrame(valid_confusion)), "validation")
        conf_artifact.add(wandb.Table(dataframe=pd.DataFrame(test_confusion)), "testing")
        new_run.log_artifact(conf_artifact)


        ########
        ## Now, we run and save the predictions for the best loss model
        ########

        chkpt_loss = glob.glob('lightning_logs/'+checkpoint_folder_name+'/checkpoints/best_loss*')[0]
        model = model.load_from_checkpoint(chkpt_loss)
        model.eval()

        train_data = RetCCLFeatureLoaderMem(train_file_list,train_labels, patches_per_iter='all')
        train_dataloader = DataLoader(train_data, batch_size=1, num_workers=4, shuffle=False)

        valid_data = RetCCLFeatureLoaderMem(valid_file_list,valid_labels, patches_per_iter='all')
        valid_dataloader = DataLoader(valid_data, batch_size=1, num_workers=4, shuffle=False)

        test_data = RetCCLFeatureLoaderMem(test_file_list, test_labels, patches_per_iter='all')
        test_dataloader = DataLoader(test_data, batch_size=1, num_workers=4, shuffle=False)

        train_preds = [model(torch.tensor(x).to(model.device))[1].detach().item() for x,y, mask in iter(train_dataloader)]
        train_probs = [model(torch.tensor(x).to(model.device))[0].detach().item() for x,y, mask in iter(train_dataloader)]   
        train_confusion = compute_confusion_matrix(train_labels.astype(int), np.array(train_preds).astype(int))

        valid_preds = [model(torch.tensor(x).to(model.device))[1].detach().item() for x,y, mask in iter(valid_dataloader)]
        valid_probs = [model(torch.tensor(x).to(model.device))[0].detach().item() for x,y, mask in iter(valid_dataloader)]
        valid_confusion = compute_confusion_matrix(valid_labels.astype(int), np.array(valid_preds).astype(int))


        test_preds = [model(torch.tensor(x).to(model.device))[1].detach().item() for x,y, mask in iter(test_dataloader)]
        test_probs = [model(torch.tensor(x).to(model.device))[0].detach().item() for x,y, mask in iter(test_dataloader)]
        test_confusion = compute_confusion_matrix(test_labels.astype(int), np.array(test_preds).astype(int))

        pred_artifact = wandb.Artifact("preds_loss", type="predictions_loss")
        pred_artifact.add(wandb.Table(dataframe=pd.DataFrame(train_preds)), "training")
        pred_artifact.add(wandb.Table(dataframe=pd.DataFrame(valid_preds)), "validation")
        pred_artifact.add(wandb.Table(dataframe=pd.DataFrame(test_preds)), "testing")
        pred_artifact.add(wandb.Table(dataframe=pd.DataFrame(train_probs)), "training_probs")
        pred_artifact.add(wandb.Table(dataframe=pd.DataFrame(valid_probs)), "validation_probs")
        pred_artifact.add(wandb.Table(dataframe=pd.DataFrame(test_probs)), "testing_probs")
        new_run.log_artifact(pred_artifact)

        conf_artifact = wandb.Artifact("confusion_loss", type="predictions_loss")
        conf_artifact.add(wandb.Table(dataframe=pd.DataFrame(train_confusion)), "training")
        conf_artifact.add(wandb.Table(dataframe=pd.DataFrame(valid_confusion)), "validation")
        conf_artifact.add(wandb.Table(dataframe=pd.DataFrame(test_confusion)), "testing")
        new_run.log_artifact(conf_artifact)

        new_run.finish()

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument("--group_id", type=str, default="UV5E2W")

    # All other arguments will be fetched from the argument store in WandB

    # Parse the user inputs and defaults (returns a argparse.Namespace)
    args = parser.parse_args()

    ## Needed on some servers at DKFZ 
    os.environ['HTTP_PROXY']="http://www-int.dkfz-heidelberg.de:80"
    os.environ['HTTPS_PROXY']="http://www-int.dkfz-heidelberg.de:80"

    main(args)