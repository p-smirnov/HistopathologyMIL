#!/bin/bash


TCGACODE=$1
echo $TCGACODE

SLIDEFILE=/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/metadata/TCGA/slides_list/TCGA-${TCGACODE}_ffpe_primary.txt
# SLIDEFILE=/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/metadata/TCGA/test.txt

BASEPATH=/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/tiles/256/TCGA/ffpe/
OUT_PATH=/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/TCGA/ffpe/HIPT_256/


source ~/.bashrc

mamba activate HIPT

python extractHIPT4K.py $SLIDEFILE $BASEPATH $OUT_PATH