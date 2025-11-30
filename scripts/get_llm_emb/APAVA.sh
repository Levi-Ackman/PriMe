#!/bin/bash
export CUDA_VISIBLE_DEVICES=0


data="APAVA"
data_path="./dataset/APAVA/"
divides=("train" "val" "test")

for divide in "${divides[@]}"; do
    python LLMemb/store_emb.py --data $data --root_path $root_path --divide $divide 
done
