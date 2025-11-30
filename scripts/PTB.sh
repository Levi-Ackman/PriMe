#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

mkdir -p ./logs/PTB
log_dir="./logs/PTB"

model_name=PriMe
data_path="./dataset/PTB/"
data_name="PTB"

bss=(128)
lrs=(1e-4)
t_layers=(6)
c_layers=(6)
llm_layers=(1)

dropouts=(0.)
d_models=(128)
patch_lens=(1)

for bs in "${bss[@]}"; do
    for lr in "${lrs[@]}"; do
        for t_layer in "${t_layers[@]}"; do
            for c_layer in "${c_layers[@]}"; do
                for dropout in "${dropouts[@]}"; do
                    for d_model in "${d_models[@]}"; do
                        for patch_len in "${patch_lens[@]}"; do
                            for llm_layer in "${llm_layers[@]}"; do
                                python -u run.py \
                                    --root_path $data_path \
                                    --model $model_name \
                                    --data $data_name \
                                    --t_layer $t_layer \
                                    --c_layer $c_layer \
                                    --llm_layer $llm_layer\
                                    --batch_size $bs \
                                    --d_model $d_model \
                                    --dropout $dropout \
                                    --patch_len $patch_len\
                                    --augmentations none,channel0.4,drop0.2 \
                                    --init srandn \
                                    --lradj constant \
                                    --itr 5 \
                                    --learning_rate $lr \
                                    --train_epochs 80 \
                                    --patience 80 > "${log_dir}/bs${bs}_lr${lr}_tl${t_layer}_cl${c_layer}_llm_${llm_layer}_dp${dropout}_dm${d_model}_pl${patch_len}.log"
                            done
                        done
                    done
                done
            done
        done
    done
done
