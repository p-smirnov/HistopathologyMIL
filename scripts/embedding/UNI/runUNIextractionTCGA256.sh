#! /bin/bash 

source /home/p163v/.bashrc


TCGACODE=$1
echo $TCGACODE

mamba activate UNI

SLIDES_LIST=/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/metadata/TCGA/slides_list/TCGA-${TCGACODE}_ffpe_primary.txt


readarray slide_list < <(basename -s .svs $(cat $SLIDES_LIST) | sed s/\\.\.*//)

python testEncodingUNI.py --slide_names ${slide_list[@]} --output_path /home/p163v/histopathology/TCGA/ffpe/UNI/ --tile_size 256 --dataset 'TCGA/ffpe'
