gpu=$1
model=$2
bert_dir=$3
output_dir=$4

# Response Selection
CUDA_VISIBLE_DEVICES=$gpu python main.py \
    --my_model=dual_encoder_ranking \
    --do_train \
    --task=nlg \
    --task_name=rs \
    --example_type=turn \
    --model_type=${model} \
    --model_name_or_path=${bert_dir} \
    --output_dir=${output_dir}/RS/MWOZ/ \
    --batch_size=25 --eval_batch_size=100 \
    --usr_token=[USR] --sys_token=[SYS] \
    --fix_rand_seed \
    --eval_by_step=1000 \
    --max_seq_length=256 \

CUDA_VISIBLE_DEVICES=$gpu python main.py \
    --my_model=dual_encoder_ranking \
    --do_train \
    --dataset='["universal_act_dstc2"]' \
    --task=nlg --task_name=rs \
    --example_type=turn \
    --model_type=${model} \
    --model_name_or_path=${bert_dir} \
     --output_dir=${output_dir}/RS/DSTC2/ \
    --batch_size=25 --eval_batch_size=100 \
    --max_seq_length=256\
    --fix_rand_seed \

CUDA_VISIBLE_DEVICES=$gpu python main.py \
    --my_model=dual_encoder_ranking \
    --do_train \
    --dataset='[\"universal_act_sim_joint\"]' \
    --task=nlg --task_name=rs \
    --example_type=turn \
    --model_type=${model} \
    --model_name_or_path=${bert_dir} \
     --output_dir=${output_dir}/RS/SIM_JOINT/ \
    --batch_size=25 --eval_batch_size=100 \
    --max_seq_length=256 \
    --fix_rand_seed
