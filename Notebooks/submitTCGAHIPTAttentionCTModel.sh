#!/bin/bash
source /home/p163v/.bashrc

mamba activate HIPT

python TCGA-HIPTAttentionCTModel.py --hidden_dim 512 256 512 256 --attention_dim 256 256 --patches_per_pat 300
