#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import zarr
import re


path_to_h5 = '/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/UKHD_Neuro/RetCLL_Features'


# now we write out the full attention maps to zarr files
for fl in list(valid_data[0]):
    slidename = re.sub('\.h5', '',fl)
    print('Reading Per Tile Attention Map ' + slidename)
    root = zarr.open("../metadata/attention_maps"+"/"+ slidename + "_per_tile_attention.zarr", mode='r') 
    coords = root['coords'][:]
    attn = root['attn'][:]
    print('Writing Attention Map ' + slidename)


    outarray = zarr.open("../metadata/attention_maps"+"/"+ slidename + ".zarr", mode='w', 
                         shape=(coords[:,1].max()+real_patch_size*2,coords[:,0].max()+real_patch_size*2), dtype=np.float32, 
                         chunks=(3072, 3072), fill_value=0.0) # arbitrary chunking of about 4x4 patches 
start_time = time.time()
for i in range(coords.shape[0]):
    attn_nonzero = outarray[coords[i,1]:(coords[i,1]+real_patch_size*2), coords[i,0]:(coords[i,0]+real_patch_size*2)] != 0
    attn_score = attn[i]
    cur_attn = np.zeros((real_patch_size*2,real_patch_size*2))
    cur_attn[~attn_nonzero] = attn_score
    cur_attn[attn_nonzero] = (attn_score + outarray[coords[i,1]:(coords[i,1]+real_patch_size*2), coords[i,0]:(coords[i,0]+real_patch_size*2)][attn_nonzero])/2
    outarray[coords[i,1]:(coords[i,1]+real_patch_size*2), coords[i,0]:(coords[i,0]+real_patch_size*2)] = cur_attn
end_time = time.time()