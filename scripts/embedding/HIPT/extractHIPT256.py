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
# from hipt_4k import HIPT_4K
from hipt_model_utils import get_vit256, eval_transforms
import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader

batch_size = 96 ## optimized on the Titan Xp

def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


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

# TILEPATH='/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/tiles/256/TCGA/ffpe/TCGA-OR-A5J1-01Z-00-DX1/'
# OUT_PATH='/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/TCGA/ffpe/HIPT_256/'

os.makedirs(OUT_PATH, exist_ok=True)

pretrained_weights256 = '/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/software/HIPT/HIPT_4K/Checkpoints/vit256_small_dino.pth'
device256 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### loading pretrained ViT_256 
model256 = get_vit256(pretrained_weights=pretrained_weights256, device=device256)
model256 = model256.to(device256)
model256.eval()



with open(slide_file, 'r') as f:
    slide_name = f.readline()
    while slide_name: 
        slide_name = re.split('\.',slide_name)[0]
        TILEPATH = basepath + '/' + slide_name 
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

print("Done")