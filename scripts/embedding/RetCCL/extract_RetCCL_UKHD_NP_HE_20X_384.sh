#!/bin/bash

source ~/.bashrc_gpu

mamba activate marugoto

python3.9 -m marugoto.extract.xiyue_wang --checkpoint-path /omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/pretrained_models/RetCCL/best_ckpt.pth --outdir /omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/UKHD_Neuro/RetCLL_Features/299_3/ /omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/tiles/384/UKHD_NP_HE/*
