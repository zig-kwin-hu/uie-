#!/usr/bin/env bash
set -e
set -x
alias python=python3
# python3 -c "import nltk; nltk.download('punkt', quiet=True)"
BD_NAME=.
export NLTK_DATA=$BD_NAME/nltk_data

export BYTED_TORCH_FX=O0
export NCCL_IB_DISABLE=0 
export NCCL_IB_HCA=$ARNOLD_RDMA_DEVICE:1 
export NCCL_IB_GID_INDEX=3 
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# export TOKENIZERS_PARALLELISM=false
# export TF_ENABLE_LEGACY_FILESYSTEM=1
# export HF_DATASETS_OFFLINE=1 
# export TRANSFORMERS_OFFLINE=1

# model and data
model_name_or_path=google/flan-t5-xl
data_dir=$BD_NAME/data/ie_instruct
output_dir=$BD_NAME/output/flan-t5-xxl-v8-plus-aux-oversamples-newf
export TRANSFORMERS_CACHE=$BD_NAME/huggingface

# configs
DEEPSPEED_CONFIG=./configs/ds_configs/stage3.config


MASTER_ADDR=localhost
MASTER_PORT=1234

N_GPUS=${ARNOLD_WORKER_GPU}
MICRO_TRAIN_BATCH_SIZE=2
MICRO_EVAL_BATCH_SIZE=8
ACCUMULATION_STEP=16

RUN_NAME=flan-t5-xxl-z3-b2-a16

# port=$(shuf -i25000-30000 -n1)

deepspeed --master_port $MASTER_PORT src/run_uie.py \
   --do_train \
   --predict_with_generate \
   --model_name_or_path $model_name_or_path \
   --data_dir $data_dir \
   --task_config_dir ./configs/multi_task_configs \
   --instruction_file ./configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir $output_dir \
   --input_record_file flan-t5.record \
   --per_device_train_batch_size $MICRO_TRAIN_BATCH_SIZE \
   --per_device_eval_batch_size $MICRO_EVAL_BATCH_SIZE \
   --gradient_accumulation_steps $ACCUMULATION_STEP \
   --learning_rate 1e-05 \
   --num_train_epochs 5 \
   --deepspeed $DEEPSPEED_CONFIG \
   --run_name $RUN_NAME \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --max_num_instances_per_task 10000 \
   --max_num_instances_per_eval_task 200 \
   --add_task_name False \
   --add_dataset_name False \
   --num_examples 0 \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 500 \
   --evaluation_strategy no \
   --save_strategy steps \
   --save_steps 1000
