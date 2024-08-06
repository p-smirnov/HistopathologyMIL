#!/bin/bash

source ~/.bashrc_gpu

mamba activate marugoto

python3.9 -m marugoto.extract.xiyue_wang --checkpoint-path /omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/pretrained_models/RetCCL/best_ckpt.pth --outdir /omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/TCGA/ffpe/1024 /omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/tiles/1024/TCGA/ffpe/*
