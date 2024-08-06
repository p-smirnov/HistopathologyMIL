#!/bin/bash


TCGACODE=$1
echo $TCGACODE

SLIDEFILE=/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/metadata/TCGA/slides_list/TCGA-${TCGACODE}_ffpe_primary.txt
# SLIDEFILE=/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/metadata/TCGA/test.txt

BASEPATH=/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/tiles/4096/TCGA/ffpe/
OUT_PATH=/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/TCGA/ffpe/HIPT_4K/


source ~/.bashrc

mamba activate HIPT

python extractHIPT256.py $SLIDEFILE $BASEPATH $OUT_PATH