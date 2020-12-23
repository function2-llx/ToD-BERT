#!/bin/bash

gpu=$1
model=$2
bert_dir=$3
output_dir=$4
bsz=32

# ./scripts/run_intent.sh 0 bert bert-base-uncased save/BERT
# ./scripts/run_intent.sh 1 todbert TODBERT/TOD-BERT-MLM-V1 save/TOD-BERT-mlm
# ./scripts/run_intent.sh 2 todbert TODBERT/TOD-BERT-JNT-V1 save/TOD-BERT-jnt

# Intent
CUDA_VISIBLE_DEVICES=$gpu python main.py \
    --my_model=multi_class_classifier \
    --dataset='["oos_intent"]' \
    --task_name="intent" \
    --earlystop="acc" \
    --output_dir=${output_dir}/Intent/OOS \
    --do_train \
    --task=nlu \
    --example_type=turn \
    --model_type=${model} \
    --model_name_or_path=${bert_dir} \
    --batch_size=${bsz} \
    --usr_token=[USR] --sys_token=[SYS] \
    --epoch=50 --eval_by_step=500 --warmup_steps=250 \
    --fix_rand_seed --nb_runs 3

