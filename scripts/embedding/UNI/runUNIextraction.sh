#! /bin/bash 

source /home/p163v/.bashrc

mamba activate UNI

readarray unprocessed_slides < /home/p163v/histopathology/metadata/UNI_384_unextracted.txt

python testEncodingUNI.py --slide_names ${unprocessed_slides[@]}
