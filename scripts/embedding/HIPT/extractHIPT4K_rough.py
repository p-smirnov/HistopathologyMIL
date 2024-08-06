#!/usr/bin/python

import sys
sys.path.insert(1,'/home/p163v/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/software/HIPT/HIPT_4K')
import os
import pickle

import glob
import re


import numpy as np
import zarr
from PIL import Image
import tqdm
from hipt_4k import HIPT_4K
from hipt_model_utils import get_vit4k, get_vit256
from hipt_4k import HIPT_4K
from hipt_model_utils import eval_transforms
import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader

batch_size = 96 ## optimized on the Titan Xp

def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


# class HIPT256(Dataset):
#     def __init__(self, zarrpath, coordsfile):
#         # self.tilepath = tilepath
#         # alltiles = glob.glob(self.tilepath + '/*.jpg')
#         # alltiles.sort()
#         self.zarr = zarr.open(zarrpath, mode='r')
#         self.coords = np.array(pickle.load(open(coordsfile, "rb")))
#     def __len__(self):
#         return self.zaar.shape[0]
#     def __getitem__(self, idx):
#         return im_tens



class TileDS(Dataset):
    def __init__(self, alltiles):
        # self.tilepath = tilepath
        # alltiles = glob.glob(self.tilepath + '/*.jpg')
        # alltiles.sort()
        self.tiles = alltiles
    def __len__(self):
        return len(self.tiles)
    def __getitem__(self, idx):
        im = Image.open(self.tiles[idx])
        im_tens = eval_transforms()(im)
        im.close()
        return im_tens

slide_file = sys.argv[1]

basepath=sys.argv[2]
OUT_PATH=sys.argv[3]

# TILEPATH='/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/tiles/4096/TCGA/ffpe/TCGA-02-0003-01Z-00-DX1'
# OUT_PATH='/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/TCGA/ffpe/HIPT_4K/'

os.makedirs(OUT_PATH, exist_ok=True)

devicemodel = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

alltiles = glob.glob(TILEPATH + '/*.png')

model = HIPT_4K()
model.eval()


with open(slide_file, 'r') as f:
    slide_name = f.readline()
    while slide_name: 
    coords = [re.split('_', re.sub('\.png', '',os.path.basename(x)))[1:3] for x in alltiles]
    coords = [(int(x),int(y)) for x,y in coords]
    sort_order = argsort(coords)
    coords = [coords[i] for i in sort_order]
    alltiles = [alltiles[i] for i in sort_order]

    os.makedirs(OUT_PATH, exist_ok=True)
    outarray = zarr.open(OUT_PATH+"/"+  slide_name + ".zarr", mode='w', shape=(len(alltiles), 192), dtype=np.float32, chunks=(1, 192))

    tileDataset = TileDS(alltiles)
    infer_dataloader = DataLoader(tileDataset, batch_size=1, num_workers=1, shuffle=False)

    i = 0
    for (batch_idx, data) in tqdm.tqdm(enumerate(infer_dataloader)):
        x = data.to(devicemodel)
        cur_batch_size = x.shape[0]
        y = model(x)
        y = y.detach().cpu().squeeze().numpy()
        outarray[batch_idx*cur_batch_size:(batch_idx+1)*cur_batch_size,:] = y.reshape(-1,192)
        i = i + cur_batch_size

    coordfile = open(OUT_PATH+"/"+  slide_name + ".coords", "wb")
    pickle.dump(coords, coordfile)
    coordfile.close()
    print('Done slide: ' + slide_name + '\n')
    slide_name = f.readline()
print("Done")



empty_tensor = eval_transforms()(np.zeros([256,256,3], dtype=np.float32)).reshape([1,3,256,256]).to(device256)
empty_embedding = model256(empty_tensor).detach().cpu().squeeze().numpy()


zarrpath = basepath + '/' + slide_name + '.zarr'
coordsfile = basepath + '/' + slide_name + '.coords'
zarr = zarr.open(zarrpath, mode='r')
coords = pickle.load(open(coordsfile, "rb"))
coords_arr = np.array(coords)

## Adopting the same strategy of center crops as HIPT codebase

make_divisble = lambda l, patch_size: (l - (l % patch_size))

x_range = np.array([coords_arr[:,0].min(), coords_arr[:,0].max()])/256
y_range = np.array([coords_arr[:,1].min(), coords_arr[:,1].max()])/256

features = np.zeros((int(x_range[1]-x_range[0]),int(y_range[1]-y_range[0]), 384), dtype=np.float32) + empty_embedding

for i in tqdm.tqdm(range((int(x_range[1]-x_range[0])))):
    for j in range(int(y_range[1]-y_range[0])):
        x = int(x_range[0] + i)
        y = int(y_range[0] + j)
        if (x*256,y*256) in coords:
            features[i,j,:] = zarr[coords.index((x*256,y*256)),:]



target_x_width = make_divisble(x_range[1] - x_range[0], 4096)
target_y_width = make_divisble(y_range[1] - y_range[0], 4096)


with open(slide_file, 'r') as f:
    slide_name = f.readline()
    while slide_name: 
        slide_name = re.split('\.',slide_name)[0]
        zarrpath = basepath + '/' + slide_name + '.zarr'
        coordsfile = basepath + '/' + slide_name + '.coords'
        print('Processing slide: ' + slide_name + '\n')

        ### Getting all tiles from the slide

        alltiles = glob.glob(TILEPATH + '/*.png')


        coords = [re.split('_', re.sub('\.png', '',os.path.basename(x)))[1:3] for x in alltiles]
        coords = [(int(x),int(y)) for x,y in coords]
        sort_order = argsort(coords)
        coords = [coords[i] for i in sort_order]
        alltiles = [alltiles[i] for i in sort_order]

        os.makedirs(OUT_PATH, exist_ok=True)
        outarray = zarr.open(OUT_PATH+"/"+  slide_name + ".zarr", mode='w', shape=(len(alltiles), 384), dtype=np.float32, chunks=(1, 384))

        tileDataset = TileDS(alltiles)
        infer_dataloader = DataLoader(tileDataset, batch_size=batch_size, num_workers=8, shuffle=False)

        i = 0
        for(batch_idx, data) in tqdm.tqdm(enumerate(infer_dataloader)):
            x = data.to(device256)
            cur_batch_size = x.shape[0]
            y = model256(x)
            y = y.detach().cpu().squeeze().numpy()
            outarray[batch_idx*cur_batch_size:(batch_idx+1)*cur_batch_size,:] = y.reshape(-1,384)
            i = i + cur_batch_size

        coordfile = open(OUT_PATH+"/"+  slide_name + ".coords", "wb")
        pickle.dump(coords, coordfile)
        coordfile.close()
        print('Done slide: ' + slide_name + '\n')
        slide_name = f.readline()





## 1.6s per batch on GPU with 128 tiles per batch


# for file in tqdm.tqdm(alltiles[0:100]):
#     with Image.open(TILEPATH + '/' + file) as im:
#         x = eval_transforms()(im).unsqueeze(0).to(device256)
#         y = model25im6(x)
#         y = y.detach().cpu().squeeze().numpy()
#         outarray[i,:] = y
#         i = i + 1

