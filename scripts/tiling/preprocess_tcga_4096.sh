#!/bin/bash 

TCGACODE=$1
echo $TCGACODE
SIZE=4096

IN_DIR=/omics/odcf/analysis/OE0606_projects/pancancer_histopathology/data/TCGA/ffpe/TCGA-$TCGACODE/primary/
LOG_DIR=/home/p163v/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/logs/TCGA-$TCGACODE
OUT_DIR=/home/p163v/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/tiles/$SIZE/TCGA/ffpe
SLIDES_LIST=/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/metadata/TCGA/slides_list/TCGA-${TCGACODE}_ffpe_primary.txt

mkdir -p $LOG_DIR
mkdir -p $OUT_DIR


while read f; do
      sample_id="${f%%.*}"
      newdir="${OUT_DIR}/${sample_id}"
      if [ ! -d $newdir ]; then
            mkdir -p $newdir
            echo "Submitting tiling for $f..."
            bsub -R "rusage[mem=8GB]" -q verylong -M 8GB -o $LOG_DIR/${sample_id}_prep.out -e $LOG_DIR/${sample_id}_prep.err -J preproc "python preprocess_slides.py $IN_DIR/$f $newdir/ $SIZE"
      fi
done < $SLIDES_LIST
echo "Done."


#while read f; do
#    cd /hps/research/gerstung/dhp/data/tcga/wsi/frozen/$f/
#    ls > /hps/research/gerstung/patel/img_preprocess/data/${f}_frozen.txt
#    done < $CANCER_LIST
