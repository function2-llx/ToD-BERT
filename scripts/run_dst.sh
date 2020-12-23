#!/bin/bash

# ./run_dst.sh 0 bert bert-base-uncased save/BERT
# ./run_dst.sh 0 todbert TODBERT/TOD-BERT-MLM-V1 save/TOD-BERT-mlm
# ./run_dst.sh 0 todbert TODBERT/TOD-BERT-JNT-V1 save/TOD-BERT-jnt

gpu=$1
model=$2
bert_dir=$3
output_dir=$4

## DST
CUDA_VISIBLE_DEVICES=$gpu python main.py \
    --my_model=BeliefTracker \
    --model_type=${model} \
    --dataset='["multiwoz"]' \
    --task_name="dst" \
    --earlystop="joint_acc" \
    --output_dir=${output_dir}/DST/MWOZ \
    --do_train \
    --task=dst \
    --example_type=turn \
    --model_name_or_path=${bert_dir} \
    --batch_size=6 --eval_batch_size=6 \
    --usr_token=[USR] --sys_token=[SYS] \
    --eval_by_step=4000 \
    --fix_rand_seed --nb_runs 10
