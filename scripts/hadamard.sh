#!/bin/bash

for SX in $(seq 0 15); do
    for SY in $(seq 0 15); do
        python main.py hadamard --batch_size 1000 \
                                --sx $SX \
                                --sy $SY \
                                --epochs 200 \
                                --num_samples 10000 \
                                --config_path configs/1x16x16-iddpm.json \
                                --save_path "hadamard/${SX}_${SY}.pt" \
                                --repeat 5
    done
done
