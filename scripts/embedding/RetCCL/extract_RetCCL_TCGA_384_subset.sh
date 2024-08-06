#!/bin/bash


TCGACODE=GBM
echo $TCGACODE

## TODO make this substitution work

arr=($(cat /home/p163v/histopathology/metadata/TCGA/slides_list/TCGA-GBM_ffpe_primary.txt | awk -F "." '{print $1}' | xargs -I{} echo /omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/tiles/384/TCGA/ffpe/{}))

source ~/.bashrc_gpu

mamba activate marugoto

python3.9 -m marugoto.extract.xiyue_wang --checkpoint-path /omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/pretrained_models/RetCCL/best_ckpt.pth --outdir /omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/TCGA/ffpe/299 ${arr[@]}
