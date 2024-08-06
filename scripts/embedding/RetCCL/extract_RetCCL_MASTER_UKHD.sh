#!/bin/bash

source ~/.bashrc_gpu

mamba activate marugoto

python3.9 -m marugoto.extract.xiyue_wang --checkpoint-path /omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/pretrained_models/RetCCL/best_ckpt.pth --outdir /omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/MASTER/UKHD_1/299 /omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/tiles/384/MASTER_UKHD_1/*
