#!/bin/bash
source /home/p163v/.bashrc

mamba activate marugoto

python /omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/Notebooks/TCGA-SimpleAttentionCTModel_CV.py --hidden_dim 1024 512 1024 512 --attention_dim 512 256 --patch_size 299 --patches_per_pat 250 --lr=0.00005