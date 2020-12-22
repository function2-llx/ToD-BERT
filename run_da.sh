gpu=$1
model=$2
bert_dir=$3
output_dir=$4

# ./run_da.sh 0 bert bert-base-uncased save/BERT
# ./run_da.sh 1 todbert TODBERT/TOD-BERT-MLM-V1 save/TOD-BERT-mlm
# ./run_da.sh 2 todbert TODBERT/TOD-BERT-JNT-V1 save/TOD-BERT-jnt

# DA
CUDA_VISIBLE_DEVICES=$gpu python main.py \
    --my_model=multi_label_classifier \
    --do_train --dataset='["multiwoz"]' \
    --task=dm --task_name=sysact --example_type=turn \
    --model_type=${model} \
    --model_name_or_path=${bert_dir} \
    --output_dir=${output_dir}/DA/MWOZ/ \
    --batch_size=8 \
    --eval_batch_size=4 \
    --learning_rate=5e-5 \
    --eval_by_step=1000 \
    --usr_token=[USR] --sys_token=[SYS] \
    --earlystop=f1_weighted \
    --fix_rand_seed --nb_runs 3

CUDA_VISIBLE_DEVICES=$gpu python main.py \
    --my_model=multi_label_classifier \
    --do_train \
    --dataset='["universal_act_dstc2"]' \
    --task=dm --task_name=sysact --example_type=turn \
    --model_type=${model} --model_name_or_path=${bert_dir} \
    --output_dir=${output_dir}/DA/DSTC2/ \
    --batch_size=8 \
    --eval_batch_size=4 \
    --learning_rate=5e-5 \
    --eval_by_step=500 \
    --usr_token=[USR] --sys_token=[SYS] \
    --earlystop=f1_weighted \
    --fix_rand_seed --nb_runs 3

CUDA_VISIBLE_DEVICES=$gpu python main.py \
    --my_model=multi_label_classifier \
    --do_train \
    --dataset='["universal_act_sim_joint"]' \
    --task=dm --task_name=sysact --example_type=turn \
    --model_type=${model} --model_name_or_path=${bert_dir} \
    --output_dir=${output_dir}/DA/SIM_JOINT \
    --batch_size=8 \
    --eval_batch_size=4 \
    --learning_rate=5e-5 \
    --eval_by_step=500 \
    --usr_token=[USR] --sys_token=[SYS] \
    --earlystop=f1_weighted \
    --fix_rand_seed --nb_runs 3
