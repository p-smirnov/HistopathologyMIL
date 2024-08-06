#!/bin/bash 


IN_DIR=/omics/odcf/analysis/OE0606_projects/pancancer_histopathology/data/UKHD_NP_HE
LOG_DIR=/home/p163v/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/logs/UKHD_NP_HE
OUT_DIR=/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/tiles/1024/UKHD_NP_HE
SLIDES_LIST=/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/UKHD_Neuro/slides_unfinished.txt

mkdir -p $LOG_DIR
mkdir -p $OUT_DIR


while read f; do
      echo "Submitting tiling for $f..."
      sample_id="${f%%.*}"
      newdir="${OUT_DIR}/${sample_id}"
      mkdir -p $newdir
      bsub -R "rusage[mem=8GB]" -q verylong -M 8GB -o $LOG_DIR/${sample_id}_prep.out -e $LOG_DIR/${sample_id}_prep.err -J preproc "python preprocess_slides.py $IN_DIR/$f $newdir/"
done < $SLIDES_LIST
echo "Done."


#while read f; do
#    cd /hps/research/gerstung/dhp/data/tcga/wsi/frozen/$f/
#    ls > /hps/research/gerstung/patel/img_preprocess/data/${f}_frozen.txt
#    done < $CANCER_LIST
