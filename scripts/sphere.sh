#!/bin/bash

python geometry.py --config_path configs/1x16x16-dit.json --save_path geometries/1x16x16-dit-ps2.pt --patch_size 2

python main.py sphere --batch_size 1000 \
                      --idx1 0 \
                      --idx2 1 \
                      --idx3 2 \
                      --epochs 1000 \
                      --num_samples 10000 \
                      --config_path configs/1x16x16-dit.json \
                      --repeat 5 \
                      --save_path "sphere/first.pt" \
                      --save_samples \
                      --network_geometry_path "geometries/1x16x16-dit-ps2.pt" \
                      --patch_size 2

python main.py sphere --batch_size 1000 \
                      --idx1 253 \
                      --idx2 254 \
                      --idx3 255 \
                      --epochs 1000 \
                      --num_samples 10000 \
                      --config_path configs/1x16x16-dit.json \
                      --repeat 5 \
                      --save_path "sphere/last.pt" \
                      --save_samples \
                      --network_geometry_path "geometries/1x16x16-dit-ps2.pt" \
                      --patch_size 2
