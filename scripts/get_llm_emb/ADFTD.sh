#!/bin/bash
export CUDA_VISIBLE_DEVICES=0


data="ADFTD"
data_path="./dataset/ADFTD/"
divides=("train" "val" "test")

for divide in "${divides[@]}"; do
    python LLMemb/store_emb.py --data $data --root_path $root_path --divide $divide 
done
