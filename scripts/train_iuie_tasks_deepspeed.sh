#!/bin/bash
declare -A epoch_map
epoch_map=([semval-RE]=10 [NYT11]=10 [SciERC]=20 [NYT11_semval-RE]=20 [NYT11_SciERC]=20 [ADE_corpus-1500]=25 [ADE_NYT11]=25 [ADE_SciERC]=25 [ADE_semval-RE]=25 [semval-RE_SciERC]=20 [SciERC_NYT11]=20 [SciERC_ADE]=25 [4combined]=20 [semval-RE_ADE]=25 [semval-RE_NYT11]=20 [SciERC_semval-RE]=20)

# declare -A TASK2DATASETS=([re]="conll04 SciERC NYT11 semval-RE ADE_corpus-1500" [eet]="ace phee casie" [eea]="ace phee casie" [ner]="CoNLL_2003 ACE_2004 ACE_2005")
# DONE : [ner] = ACE_2004 ACE_2005 AnatEM bc2gm bc4chemd bc5cdr Broad_Tweet_Corpus CoNLL_2003 FabNER FindVehicle GENIA_NER HarveyNER mit-movie mit-restaurant MultiNERD ncbi Ontonotes_sample_30000 PolyglotNER TweetNER7_sample_15000 WikiANN_en WikiNeural
#declare -A TASK2DATASETS=([re]="ADE_corpus NYT11_sample_30000 New-York-Times-RE_sample_30000 semval-RE conll04 GIDS SciERC kbp37" [eet]="ace phee casie" [eea]="ace phee casie" [ner]="ACE_2004 ACE_2005 AnatEM bc2gm bc4chemd bc5cdr Broad_Tweet_Corpus CoNLL_2003 FabNER FindVehicle GENIA_NER HarveyNER mit-movie mit-restaurant MultiNERD ncbi Ontonotes_sample_30000 PolyglotNER TweetNER7 WikiANN_en WikiNeural")
declare -A TASK2DATASETS=([re]="SciERC" [eet]="ace phee casie" [eea]="ace phee casie" [ner]="ACE_2004 ACE_2005 AnatEM bc2gm bc4chemd bc5cdr Broad_Tweet_Corpus CoNLL_2003 FabNER FindVehicle GENIA_NER HarveyNER mit-movie mit-restaurant MultiNERD ncbi Ontonotes_sample_30000 PolyglotNER TweetNER7 WikiANN_en WikiNeural")

set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=./huggingface

port=$(shuf -i25000-30000 -n1)

model_name_or_path=ZWK/InstructUIE
# model_name_or_path=google/flan-t5-xxl

# for TASK in re ner eet eea 
for TASK_CONFIG in re
do
    for DATASET_CONFIG in ${TASK2DATASETS[${TASK_CONFIG}]}
    do
        deepspeed --include localhost:0,1,2,3 --master_port $port src/run_uie.py \
        --do_train \
        --do_eval \
        --do_predict \
        --deepspeed ./configs/ds_configs/stage0.config \
        --predict_with_generate \
        --model_name_or_path ${model_name_or_path} \
        --data_dir ./data/ie_instruct \
        --task_config_dir ./configs/${TASK_CONFIG}_configs/${DATASET_CONFIG} \
        --instruction_file ./configs/instruction_config.json \
        --prompt_file ./prompts/instructUIE.json \
        --instruction_strategy multiple \
        --min_negative_labels -1 \
        --min_positive_labels -1 \
        --output_dir .output/${TASK_CONFIG}_lora_test/${DATASET_CONFIG}/iuie-xxl \
        --input_record_file iuie.record \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 8 \
        --learning_rate 5e-05 \
        --num_train_epochs 10 \
        --run_name ${model_name_or_path}-${TASK_CONFIG}-${DATASET_CONFIG} \
        --max_source_length 512 \
        --max_target_length 50 \
        --generation_max_length 50 \
        --max_num_instances_per_task 15000 \
        --max_num_instances_per_eval_task 15000 \
        --add_task_name False \
        --add_dataset_name False \
        --num_examples 0 \
        --overwrite_cache \
        --lr_scheduler_type constant \
        --warmup_step 0 \
        --logging_strategy steps \
        --logging_steps 10 \
        --evaluation_strategy epoch \
        --save_strategy epoch \
        --cache_dir ./huggingface \
        --ddp_find_unused_parameters False \
        --save_total_limit 1 \
        --over_sampling False \
        --bf16 \
        --load_best_model_at_end False \
        --metric_for_best_model eval_f1 \
        --only_save_best_model True \
        --lora_target_modules q,v \
        --lora_r 16 \
        --test_with_eval \
        --save_lora_weights_only \
        --predict_each_dataset_with_best False \
        #--overwrite_output_dir \
    done
done