import os
import numpy as np
import h5py
import click
import torch
import torch.nn as nn
import glob
import pyarrow.parquet as pq
from timm.models.vision_transformer import VisionTransformer
from torchvision import transforms
from ctran import ctranspath
from torch.utils.data import Dataset
from PIL import Image



mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
trnsfrms_val = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]
)


def get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "DINO_p16": "dino_vit_small_patch16_ep200.torch",
        "DINO_p8": "dino_vit_small_patch8_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url


def vit_small(pretrained, progress, key, **kwargs):
    patch_size = kwargs.get("patch_size", 16)
    model = VisionTransformer(
        img_size=224, patch_size=patch_size, embed_dim=384, num_heads=6, num_classes=0
    )
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        verbose = model.load_state_dict(
            torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        )
        print(verbose)
    return model


def save_hdf5(output_path, asset_dict, attr_dict=None, mode='w'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path


def comma_separated(values):
    return values.split(',')


class roi_dataset(Dataset):
    def __init__(self, slide_ls):
        super().__init__()
        self.slide_ls = slide_ls
        self.tile_ls = []
        for slide in self.slide_ls:
            self.tile_ls.extend(glob.glob(os.path.join(slide, '*.jpg')))
        self.transform = trnsfrms_val

    def __len__(self):
        return len(self.tile_ls)

    def __getitem__(self, idx):
        slide_id = self.tile_ls[idx].split('/')[-2]
        image = Image.open(self.tile_ls[idx]).convert('RGB')
        image = self.transform(image)
        spatial_x = int(self.tile_ls[idx].split('/')[-1].split('_')[-2])
        spatial_y = int(self.tile_ls[idx].split('/')[-1].split('_')[-1].split('.')[0])
        return image, slide_id, spatial_x, spatial_y


@click.command()
@click.option('--ckpt', type=str, help='path to the save directory')
@click.option('--feature_dir', type=str, help='path to the save directory')
@click.option('--split', type=str, help='path to the save directory')
def inference(split, ckpt, feature_dir):
    slide_ls = split.split(',')
    # label_filepath = pq.read_table('/dkfz/cluster/gpu/data/OE0606/histopathology/UCL/tile_info_UCL.parquet').to_pandas()
    test_datat=roi_dataset(slide_ls)
    database_loader = torch.utils.data.DataLoader(test_datat, batch_size=768, shuffle=False)

    ### change the model and load the ckpt file here ###
    model = ctranspath().cuda()
    model.head = nn.Identity()
    td = torch.load(ckpt)['state_dict']  # ./ckpt/checkpoint_056.pth.tar
    ckpt = {key[len('module.momentum_encoder.'):]: value for key, value in td.items() if 'momentum_encoder' in key and 'head' not in key}  # 匹配moco键值
    model.load_state_dict(ckpt, strict=True)
    # model = vit_small(pretrained=False, progress=False, key="DINO_p16", patch_size=16)
    # td = torch.load(ckpt)['teacher']
    # ckpt = {key[len('backbone.'):]: value for key, value in td.items() if 'backbone' in key}
    # model.load_state_dict(ckpt)
    # model.cuda()
    # database_loader = torch.utils.data.DataLoader(test_datat, batch_size=768*4, shuffle=False)  # lunit DINO直接上768*4

    ### change the model and load the ckpt file here ###
      
    model.eval()
    count = 0
    print('Inference begins...')
    with torch.no_grad():
        for batch, slide_id, spatial_x, spatial_y in database_loader:
            print(f'{count}/{len(database_loader)}')
            batch = batch.cuda()

            features = model(batch)
            features = features.cpu().numpy()
            id_set = list(np.unique(np.array(slide_id)))
            spatial_x = np.array(spatial_x)
            spatial_y = np.array(spatial_y)
            for id in id_set:
                feature = features[np.array(slide_id)==id]
                pos_x = spatial_x[np.array(slide_id)==id]
                pos_y = spatial_y[np.array(slide_id)==id]
                output_path = os.path.join(feature_dir, 'h5_files', id+'.h5')
                asset_dict = {'features': feature, 'pos_x': pos_x, 'pos_y': pos_y}
                save_hdf5(output_path, asset_dict, attr_dict=None, mode='a')
            count += 1

    h5_ls = [os.path.join(feature_dir, 'h5_files', item.split('/')[-1]) for item in slide_ls]
    os.makedirs(os.path.join(feature_dir, 'pt_files'), exist_ok=True)
    for idx, h5file in enumerate(h5_ls):
        if os.path.exists(os.path.join(feature_dir, 'pt_files', os.path.basename(h5file)+'.pt')):
            pass
        else:
            file = h5py.File(h5file+'.h5', "r")
            features = file['features'][:]
            features = torch.from_numpy(features)
            torch.save(features, os.path.join(feature_dir, 'pt_files', os.path.basename(h5file)+'.pt'))


if __name__ == '__main__':
    inference()