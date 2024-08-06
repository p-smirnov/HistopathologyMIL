#!/bin/bash

#bsub -R "rusage[mem=200GB]" -M 200GB -n 1 -R "span[hosts=1]" -W 72:00 -J tcgagbmval /bin/bash submitTCGAGBMValidatite.sh

source /home/p163v/.bashrc

mamba activate marugoto

python /omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/src/inference/Validate_UNI_TCGA_GBM.py