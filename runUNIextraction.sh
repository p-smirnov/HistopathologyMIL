#! /bin/bash 

source /home/p163v/.bashrc

mamba activate UNI

python testEncodingUNI.py --slide_names $(ls /home/p163v/histopathology/tiles/384/UKHD_NP_HE | tail -n +1653)
