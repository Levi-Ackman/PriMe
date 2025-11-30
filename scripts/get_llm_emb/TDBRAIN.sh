#!/bin/bash
export CUDA_VISIBLE_DEVICES=0


data="TDBRAIN"
data_path="./dataset/TDBRAIN/"
divides=("train" "val" "test")

for divide in "${divides[@]}"; do
    python LLMemb/store_emb.py --data $data --root_path $root_path --divide $divide 
done
