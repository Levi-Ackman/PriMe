#!/bin/bash
export CUDA_VISIBLE_DEVICES=0


data="MIMIC"
data_path="./dataset/MIMIC/"
divides=("train" "val" "test")

for divide in "${divides[@]}"; do
    python LLMemb/store_emb.py --data $data --root_path $root_path --divide $divide 
done
