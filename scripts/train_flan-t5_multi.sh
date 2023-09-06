#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=./huggingface

port=$(shuf -i25000-30000 -n1)

model_name_or_path=google/flan-t5-xl

# for TASK_CONFIG in comparison_configs/ace comparison_configs/phee comparison_configs/casie
for TASK_CONFIG in general_task_configs
do
    python src/run_uie.py \
    --do_train \
    --do_predict \
    --resume_from_checkpoint ./output/${TASK_CONFIG}/flan-t5-xl_256bs/checkpoint-1125 \
    --predict_with_generate \
    --model_name_or_path ${model_name_or_path} \
    --data_dir ./data/ie_instruct \
    --task_config_dir ./configs/${TASK_CONFIG} \
    --instruction_file ./configs/instruction_config.json \
    --prompt_file ./prompts/instructUIE.json \
    --instruction_strategy multiple \
    --min_negative_labels 5 \
    --min_positive_labels 1 \
    --ordered_prompt False \
    --output_dir ./output/${TASK_CONFIG}/flan-t5-xl_256bs \
    --input_record_file flan-t5.record \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 32 \
    --learning_rate 5e-05 \
    --num_train_epochs 5 \
    --run_name ${model_name_or_path}-ie-instruct \
    --max_source_length 512 \
    --max_target_length 50 \
    --bf16 \
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
    --logging_steps 10 \
    --evaluation_strategy no \
    --save_strategy epoch \
    --cache_dir ./huggingface \
    --ddp_find_unused_parameters False \
    --save_total_limit 1 \
    --save_steps 500 \
    --over_sampling False \

done

