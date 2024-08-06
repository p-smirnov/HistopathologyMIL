#!/bin/bash 

TCGACODE=$1
echo $TCGACODE


IN_DIR=/omics/odcf/analysis/OE0606_projects/pancancer_histopathology/data/TCGA/ffpe/TCGA-$TCGACODE/primary/
LOG_DIR=/home/p163v/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/logs/CellVIT_tiling/TCGA-$TCGACODE
SLIDES_LIST=/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/metadata/TCGA/slides_list/TCGA-${TCGACODE}_ffpe_primary.txt
OUT_DIR=/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/tiles/1024/TCGA/CellVIT/
CODE_DIR=/home/p163v/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/software/CellViT/preprocessing/patch_extraction/

mkdir -p $LOG_DIR
mkdir -p $OUT_DIR

var=0
while read f; do
    if [ -d $OUT_DIR/$(basename -s .svs $f) ] ; then
      continue
    fi
    echo "Submitting tiling for $f..."
    sample_id="${f%%.*}"
    # newdir="${OUT_DIR}/${sample_id}"
    # mkdir -p $newdir
    bsub -R "rusage[mem=36GB]" -M 36GB -n 1 -R "span[hosts=1]" -W 2:00 -o $LOG_DIR/${sample_id}_prep.out -e $LOG_DIR/${sample_id}_prep.err -J cellvit_tile "python $CODE_DIR/main_extraction.py --wsi_paths $IN_DIR/$f --output_path $OUT_DIR --patch_size 1024 --patch_overlap 6.25 --target_mag 40 --processes 1 --log_path $LOG_DIR/${sample_id}_prep_log"
    var=$((var+1))
    # if [ $var -eq 100 ]; then
    #       break
    # fi
    # if [ -d $OUT_DIR/$(basename -s .svs $f) ] ; then
    #   bsub -R "rusage[mem=1GB]" rm -rf $OUT_DIR/$(basename -s .svs $f)
    # fi
done < $SLIDES_LIST
echo "Done."


#while read f; do
#    cd /hps/research/gerstung/dhp/data/tcga/wsi/frozen/$f/
#    ls > /hps/research/gerstung/patel/img_preprocess/data/${f}_frozen.txt
#    done < $CANCER_LIST
