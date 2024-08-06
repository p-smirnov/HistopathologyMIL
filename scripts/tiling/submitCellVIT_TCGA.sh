#!/bin/bash 
TCGACODE=SARC
CODE_DIR=/home/p163v/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/software/CellViT/cell_segmentation/inference
PATCH_DIR=/home/p163v/histopathology/tiles/1024/TCGA/CellVIT/
IN_DIR=/omics/odcf/analysis/OE0606_projects/pancancer_histopathology/data/TCGA/ffpe/TCGA-$TCGACODE/primary/

SLIDE="TCGA-3B-A9HL-01Z-00-DX1.E7B32155-5633-4623-A8F9-D51F0EB4E8EA"


bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=37G -q gpu -R "rusage[mem=80GB]" -M 80GB -n 1 -R "span[hosts=1]" -W 26:00 -J cellvit "python $CODE_DIR/cell_detection.py --model /home/p163v/histopathology/pretrained_models/CellVIT/CellViT-SAM-H-x40.pth --geojson process_wsi --wsi_path $IN_DIR/$SLIDE.svs --patched_slide_path $PATCH_DIR/$SLIDE"
