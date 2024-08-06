#!/bin/bash

#bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=20G -q gpu -R "rusage[mem=200GB]" -M 200GB -n 1 -R "span[hosts=1]" -W 72:00 -J ukhdretccl /bin/bash submitRetCCL_UKHD_CV_UNI.sh

source /home/p163v/.bashrc

mamba activate marugoto

python /omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/src/train/train.py --hidden_dim 1024 512 1024 512 --attention_dim 512 256 --patch_size 256 --patches_per_pat 250 --lr=0.00005  --embedding UNI_256 --cv_split_path /omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/splits/12062024_UNI_DIFG/