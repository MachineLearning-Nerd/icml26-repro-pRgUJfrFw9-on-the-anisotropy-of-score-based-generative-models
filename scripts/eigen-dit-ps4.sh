#!/bin/bash

python geometry.py --config_path configs/1x16x16-dit.json --save_path geometries/1x16x16-dit-ps4.pt --patch_size 4

for idx in 0 31 63 95 127 159 191 223 255
do
    python main.py eigen --batch_size 1000 \
                         --idx $idx \
                         --epochs 400 \
                         --num_samples 10000 \
                         --config_path configs/1x16x16-dit.json \
                         --repeat 5 \
                         --save_path "eigen-dit-ps4/${idx}.pt" \
                         --network_geometry_path "geometries/1x16x16-dit-ps4.pt" \
                         --patch_size 4
done
