#! /bin/bash 

# bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=23G -q gpu -R "rusage[mem=200GB]" -M 200GB -n 1 -R "span[hosts=1]" -W 72:00 -J extractUNI /bin/bash runSpatialMedulloExtraction.sh

source /home/p163v/.bashrc

mamba activate UNI

# readarray unprocessed_slides < /home/p163v/histopathology/metadata/UNI_384_unextracted.txt

python testEncodingUNI.py --dataset "spatial_medullo/tiles" --tile_size "20x_256" --output_path /home/p163v/histopathology/spatial_medullo/UNI_embeddings/ --slide_names $(ls /home/p163v/histopathology/tiles/20x_256/spatial_medullo/tiles/)
