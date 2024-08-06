#!/bin/bash

#bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=20G -q gpu -R "rusage[mem=200GB]" -M 200GB -n 1 -R "span[hosts=1]" -W 72:00 -J ukhdretccl /bin/bash submitRetCCL_UKHD_CV_UNI_Transformer.sh

source /home/p163v/.bashrc

mamba activate marugoto

python /omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/Notebooks/UMAP_UKHD_Neuro-SimpleAttentionCTModel_CV.py --model TransformerMIL --hidden_dim  512 512 --n_heads 8 --patch_size 256 --patches_per_pat 250 --embedding UNI_256 --cv_split_path /omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/splits/15052024_UNI/ --lr 0.00002 --weight_decay 0.00002
