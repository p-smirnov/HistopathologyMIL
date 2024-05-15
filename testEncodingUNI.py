import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image
import torch
import os
import pickle
from torch.utils.data import Dataset, DataLoader
import re
import numpy as np
import zarr

import argparse

parser = argparse.ArgumentParser(description='Extract embeddings from UNI model')
parser.add_argument('--tile_size', type=int, default=384, help='Tile size')
parser.add_argument('--slide_names', type=str, default=['0A2F095A-1117-4F72-B648-717BAA3FD4AE'], help='Slide name', nargs="+")
parser.add_argument('--output_path', type=str, default='/home/p163v/histopathology/UKHD_Neuro/UNI_embeddings/', help='Output path')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
args = parser.parse_args()


# batch_size = 512
# tile_size = 384

output_path = args.output_path + str(args.tile_size) + '/'
# slide_name = '0A2F095A-1117-4F72-B648-717BAA3FD4AE'


# with open('/home/p163v/histopathology/tiles/384/UKHD_NP_HE/0A2F095A-1117-4F72-B648-717BAA3FD4AE/0A2F095A-1117-4F72-B648-717BAA3FD4AE_tiles_list_png.txt','rb') as fl: 
#     data = pickle.load(fl)

# img_dir = '/home/p163v/histopathology/tiles/384/UKHD_NP_HE/0A2F095A-1117-4F72-B648-717BAA3FD4AE/'
# with open(img_dir + '/' + os.path.basename(os.path.dirname(img_dir)) + '_tiles_list_png.txt','rb') as fl: 
#             tile_list = pickle.load(fl) 
# pickle.load('/home/p163v/histopathology/tiles/384/UKHD_NP_HE/0A2F095A-1117-4F72-B648-717BAA3FD4AE/0A2F095A-1117-4F72-B648-717BAA3FD4AE_1196_15548.png.txt')

class TileDataset(Dataset):
    def __init__(self, img_dir, transform=None,):
        self.img_dir = img_dir
        with open(img_dir + '/' + os.path.basename(os.path.dirname(img_dir)) + '_tiles_list_png.txt','rb') as fl: 
            self.tile_list = pickle.load(fl) 
        self.transform = transform
    def __len__(self):
        return len(self.tile_list)
    def __getitem__(self, idx):
        cur_tile = self.tile_list[idx]
        coords = re.search(r"(\d+)_(\d+)", str(cur_tile)).group().split('_') 
        img_path = os.path.join(self.img_dir, cur_tile)
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, np.array(coords).astype(int)

model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True).cuda()
transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
model.eval()

# model, transform = get_encoder(enc_name='uni', device='cpu')


for slide_name in args.slide_names:
    print(f'Processing slide {slide_name}')

    dataset = TileDataset('/home/p163v/histopathology/tiles/'+ str(args.tile_size) +'/UKHD_NP_HE/'+slide_name+'/', transform=transform)
    inference_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True) # good for a 24GB GPU, but too big for 20GB


    embeddings = np.zeros((len(dataset), 1024), dtype=np.float32)
    coords = np.zeros((len(dataset), 2), dtype=np.int32)

    for i, data in enumerate(iter(inference_dataloader)): 
        print(f'Batch {i+1}/{len(inference_dataloader)}')
        x, coord = data  
        with torch.inference_mode():
            feature_emb = model(x.cuda()).cpu() # Extracted features (torch.Tensor) with shape [args.batch_size,1024]
        embeddings[i*args.batch_size:(i+1)*args.batch_size] = feature_emb.detach().numpy()
        coords[i*args.batch_size:(i+1)*args.batch_size] = coord

    output_root = zarr.open(args.output_path +'/'+ slide_name + '_uni_embedding.zarr' , mode='w') 
    output_root['features'] = zarr.array(embeddings, dtype=np.float32, chunks=(128, 1024), compression= 'none') # approximately 500kb per chunk
    output_root['coords'] = zarr.array(coords, dtype=np.int32, compression= 'none')
    output_root.attrs['tile_size'] = args.tile_size
    output_root.attrs['slide_name'] = slide_name
    output_root.attrs['model'] = 'UNI'
    output_root.attrs['embedding_size'] = 1024
    output_root.attrs['num_tiles'] = len(dataset)

