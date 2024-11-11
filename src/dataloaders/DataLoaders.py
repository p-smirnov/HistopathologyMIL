import numpy as np
import pandas as pd
import os
import sys
import h5py
import re
import sklearn


import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch import optim, utils, Tensor
import glob 
from PIL import Image
# from turbojpeg import TurboJPEG
import pickle
import zarr
from src.utils.embedding_loaders import check_slide_exists, load_single_features


class RetCCLFeatureLoader(Dataset):
    def __init__(self, slide_filenames, feature_path, labels, patches_per_iter = 10):
        assert len(labels) == len(slide_filenames)
        self.labels = labels
        self.file_paths = [feature_path + "/" +  x for x in slide_filenames]
        self.slide_names = slide_filenames
        self.num_patches = patches_per_iter

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        cur_path = self.file_paths[idx]
        label = self.labels[idx]
        features = h5py.File(cur_path, 'r')['feats']
        # features = features.reshape([1,-1,2048])
        sampled_pchs = np.random.choice(range(features.shape[0]),size=self.num_patches, replace=False)
        sampled_pchs = np.sort(sampled_pchs)
        features = features[sampled_pchs,:]
        #if not torch.cuda.is_available():
        features = features.astype(np.float32)
        return features, label



class RetCCLFeatureLoaderMem(Dataset):
    def __init__(self, slide_list, labels, patches_per_iter = 10):
        assert len(labels) == len(slide_list)
        self.labels = labels
        self.slide_list = slide_list
        self.num_patches = patches_per_iter
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        features = self.slide_list[idx]
        label = self.labels[idx]
        # features = features.reshape([1,-1,2048])
        if self.num_patches == 'all':
            features = features[:]
            mask = np.zeros([features.shape[0]])
        else:
            if self.num_patches > features.shape[0]:
                features = features[:]
                mask = np.concatenate([np.zeros([features.shape[0]]),np.ones([self.num_patches - features.shape[0]])*(-np.inf)], axis=0)
                features = np.pad(features, [(0,self.num_patches - features.shape[0]),(0,0)], mode='constant', constant_values=0)
            else:
                sampled_pchs = np.random.choice(range(features.shape[0]),size=self.num_patches, replace=False)
                features = features[sampled_pchs,:]
                mask = np.zeros([features.shape[0]])
        #if not torch.cuda.is_available():
        features = features.astype(np.float32)
        mask = mask.astype(np.float32)
        return features, label, mask


class RetCCLFeatureLoaderMemAllPatches(Dataset):
    def __init__(self, slide_list, labels, patches_per_iter = 10):
        assert len(labels) == len(slide_list)
        self.labels = labels
        self.slide_list = slide_list
        self.num_patches = patches_per_iter
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        features = self.slide_list[idx]
        label = self.labels[idx]
        # features = features.reshape([1,-1,2048])
        # sampled_pchs = np.random.choice(range(features.shape[0]),size=self.num_patches, replace=False)
        features = features[:]
        #if not torch.cuda.is_available():
        features = features.astype(np.float32)
        return features, label


class RetCCLFeatureLoaderMemNoMIL(Dataset):
    def __init__(self, slide_list, labels):
        assert len(labels) == len(slide_list)
        self.labels = np.concatenate([np.repeat(labels[i], x.shape[0]) for i,x in enumerate(slide_list)]).astype('float')
        self.slide_list = np.concatenate(slide_list)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        features = self.slide_list[idx]
        label = self.labels[idx]
        features = features.astype(np.float32)
        return features, label


class TCGAPatchLoaderImages(Dataset):
    def __init__(self, basepath, slidelist, labels, patches_per_iter = 100):
        self.slides = slidelist
        assert all([slide in os.listdir(basepath) for slide in self.slides])
        assert len(labels) == len(self.slides)
        self.labels = labels
        self.slide_paths = [basepath + "/" +  x for x in self.slides]
        self.num_patches = patches_per_iter
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        # self.transform = Compose([ToTensor(), Normalize(mean = mean, std = std)])
        self.transform = Normalize(mean=mean, std=std)
        # self.decoder = TurboJPEG()
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        cur_path = self.slide_paths[idx]
        cur_slide = self.slides[idx]
        label = self.labels[idx]
        patch_list_path = cur_path + "/" + cur_slide + "_tiles_list.txt"
        with open(patch_list_path, 'rb') as f:
            patches = pickle.load(f)
        patches = [cur_path + "/" + x for x in patches]
        sampled_pchs = np.random.choice(range(len(patches)),size=self.num_patches, replace=False)
        cur_patches = torch.stack([self.transform(torchvision.io.read_image(patches[x]).float()) for x in sampled_pchs])
        # cur_patches = torch.stack([self.transform(Image.open(patches[x])) for x in sampled_pchs])
        return cur_patches, label

def load_turbojpeg(path, decoder):
    with open(path, "rb") as f:
        imarray = decoder.decode(f.read())
    imarray = imarray[:,:,[2,1,0]]
    imarray.transpose([2,0,1])
    return imarray



class HIPT256FeatureLoader(Dataset):
    def __init__(self, basepath, slidelist, labels, patches_per_iter = 100):
        self.slides = slidelist
        self.basepath = basepath
        assert all([slide + ".zarr" in os.listdir(basepath) for slide in self.slides])
        assert len(labels) == len(self.slides)
        self.labels = labels
        self.num_patches = patches_per_iter
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        cur_slide = zarr.open(self.basepath+"/"+  self.slides[idx] + ".zarr", mode='r')
        # cur_path = self.slide_paths[idx]
        # cur_slide = self.slides[idx]
        label = self.labels[idx]
        sampled_pchs = np.random.choice(range(cur_slide.shape[0]),size=self.num_patches, replace=False)
        cur_patches = cur_slide[sampled_pchs]
        # cur_patches = torch.stack([self.transform(Image.open(patches[x])) for x in sampled_pchs])
        return cur_patches, label


class FromDiskFeatureLoader(Dataset):
    def __init__(self, embedding, slidelist, patch_size, patches_per_iter = 'all', extra_features = None):
        self.embedding = embedding
        self.patch_size = patch_size
        self.extra_features = None
        self.extra_features = extra_features
        self.slides = [slide for slide in slidelist if check_slide_exists(embedding, patch_size, slide)]
        self.num_patches = patches_per_iter
        self.feature_size = load_single_features(self.embedding, self.patch_size, self.slides[0]).shape[1]
    def __len__(self):
        return len(self.slides)
    def __getitem__(self, idx):
        cur_features = load_single_features(self.embedding, self.patch_size, self.slides[idx])
        if self.num_patches == 'all':
            return cur_features
        sampled_pchs = np.random.choice(range(cur_features.shape[0]),size=self.num_patches, replace=False)
        cur_patches = cur_features[sampled_pchs]
        # cur_patches = torch.stack([self.transform(Image.open(patches[x])) for x in sampled_pchs])
        return cur_patches

