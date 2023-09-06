#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=./huggingface

port=$(shuf -i25000-30000 -n1)

model_name_or_path=google/flan-t5-xl

for TASK_CONFIG in SciERC
do
    CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --master_port $port src/run_uie.py \
    --deepspeed ./configs/ds_configs/hzk.config \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path ${model_name_or_path} \
    --data_dir ./data/ie_instruct \
    --task_config_dir ./configs/re_configs/${TASK_CONFIG} \
    --instruction_file ./configs/instruction_config.json \
    --prompt_file ./prompts/instructUIE.json \
    --instruction_strategy multiple \
    --min_negative_labels 5 \
    --min_positive_labels 1 \
    --output_dir ./output/${TASK_CONFIG}/flan-t5-xl_256bs\
    --input_record_file flan-t5.record \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 32 \
    --learning_rate 5e-05 \
    --num_train_epochs 30 \
    --run_name ${model_name_or_path}-RE-${TASK_CONFIG} \
    --max_source_length 512 \
    --max_target_length 50 \
    --generation_max_length 50 \
    --max_num_instances_per_task 4000 \
    --max_num_instances_per_eval_task 200 \
    --add_task_name False \
    --add_dataset_name False \
    --num_examples 0 \
    --overwrite_output_dir \
    --overwrite_cache \
    --lr_scheduler_type constant \
    --warmup_steps 0 \
    --bf16 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --cache_dir ./huggingface \
    --ddp_find_unused_parameters False \
    --save_total_limit 2 \
    --over_sampling False \
    --load_best_model_at_end True\
    --metric_for_best_model eval_f1\

done

