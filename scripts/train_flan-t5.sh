#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=./huggingface

port=$(shuf -i25000-30000 -n1)

# 最好令 bs_per_gpu * num_gpu * gradient_accumulation_steps = 256
# 学习率可以使用 5e-5
# param_num < 1b 10epoch, 3b 5epoch, 11b 5epoch
# 注意修改 CUDA_VISIBLE_DEVICES, model_name_or_path，output_dir, run_name, data_dir, task_config_dir, instruction_file
# 其余参数可与当前版本保持一致
model_name_or_path=google/flan-t5-xl

# Running tmux 
# CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --master_port $port src/run_uie.py \
#    --do_train \
#    --do_predict \
#    --predict_with_generate \
#    --model_name_or_path ${model_name_or_path} \
#    --data_dir ./data/ie_instruct \
#    --task_config_dir ./configs/multi_task_configs \
#    --instruction_file ./configs/instruction_config.json \
#    --instruction_strategy single \
#    --output_dir ./output/ie-instruct/EET+EE/flan-t5-base_256bs \
#    --input_record_file flan-t5.record \
#    --deepspeed ./configs/ds_configs/stage0.config \
#    --per_device_train_batch_size 8 \
#    --per_device_eval_batch_size 8 \
#    --gradient_accumulation_steps 8 \
#    --learning_rate 5e-05 \
#    --num_train_epochs 10 \
#    --run_name ${model_name_or_path}-ie-instruct-EET+EE \
#    --max_source_length 512 \
#    --max_target_length 50 \
#    --generation_max_length 50 \
#    --max_num_instances_per_task 10000 \
#    --max_num_instances_per_eval_task 200 \
#    --add_task_name False \
#    --add_dataset_name False \
#    --bf16 \
#    --num_examples 0 \
#    --overwrite_output_dir \
#    --overwrite_cache \
#    --lr_scheduler_type constant \
#    --warmup_steps 0 \
#    --logging_strategy steps \
#    --logging_steps 10 \
#    --evaluation_strategy no \
#    --save_strategy steps \
#    --cache_dir ./huggingface \
#    --ddp_find_unused_parameters False \
#    --save_steps 1000 \


CUDA_VISIBLE_DEVICES=0,1,2,3 python src/run_uie.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path ${model_name_or_path} \
   --data_dir ./data/ie_instruct \
   --task_config_dir ./configs/ner_task_configs \
   --instruction_file ./configs/instruction_config.json \
   --prompt_file ./prompts/instructUIE.json \
   --instruction_strategy multiple \
   --min_negative_labels 5 \
   --min_positive_labels 1 \
   --output_dir ./output/NER_other/flan-t5-xl_256bs\
   --input_record_file flan-t5.record \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 32 \
   --learning_rate 5e-05 \
   --num_train_epochs 5 \
   --run_name ${model_name_or_path}-ie-instruct \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --max_num_instances_per_task 5000 \
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
   --evaluation_strategy no \
   --save_strategy steps \
   --cache_dir ./huggingface \
   --ddp_find_unused_parameters False \
   --save_total_limit 2 \
   --save_steps 1000 \