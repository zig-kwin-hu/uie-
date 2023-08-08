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

# 3090 * 4 on t5-700M
model=meta-llama/Llama-2-7b-hf
python src/run_uie.py \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path ${model} \
   --resume_from_checkpoint ./output/fewnerd/llama2-7B_256bs/checkpoint-100 \
   --data_dir ./data/ie_instruct \
   --task_config_dir ./configs/llm_task_configs \
   --instruction_file ./configs/instruction_config.json \
   --instruction_strategy single \
   --input_record_file flan-t5.record \
   --per_device_eval_batch_size 8 \
   --run_name t5-700M-mult-mi-experiment \
   --lora_target_modules q_proj,k_proj,v_proj,o_proj \
   --max_source_length 512 \
   --max_target_length 512 \
   --generation_max_length 512 \
   --max_num_instances_per_eval_task 200 \
   --add_task_name False \
   --add_dataset_name False \
   --num_examples 0 \
   --output_dir ./output/eval/${model} \
   --overwrite_output_dir \
   --overwrite_cache \
