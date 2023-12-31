#!/bin/bash
declare -A epoch_map
epoch_map=([semval-RE]=10 [NYT11]=10 [SciERC]=20 [NYT11_semval-RE]=20 [NYT11_SciERC]=20 [ADE_corpus-1500]=25 [ADE_NYT11]=25 [ADE_SciERC]=25 [ADE_semval-RE]=25 [semval-RE_SciERC]=20 [SciERC_NYT11]=20 [SciERC_ADE]=25 [4combined]=20 [semval-RE_ADE]=25 [semval-RE_NYT11]=20 [SciERC_semval-RE]=20)

#declare -A TASK2DATASETS=([re]="ADE_corpus NYT11_sample_30000 New-York-Times-RE_sample_30000 semval-RE conll04 GIDS SciERC kbp37" [eet]="ace phee casie" [eea]="ace phee casie" [ner]="ACE_2004 ACE_2005 AnatEM bc2gm bc4chemd bc5cdr Broad_Tweet_Corpus CoNLL_2003 FabNER FindVehicle GENIA_NER HarveyNER mit-movie mit-restaurant MultiNERD ncbi Ontonotes_sample_30000 PolyglotNER TweetNER7_sample_15000 WikiANN_en WikiNeural")
declare -A TASK2DATASETS=([re]="ADE_corpus" [eet]="ace" [eea]="ace" [ner]="plo_all")

set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=./huggingface

port=$(shuf -i25000-30000 -n1)

model_name_or_path=/storage_fast/zkhu/UIE-pp/output/ner/plo_all/uie_base_fft/checkpoint-2664/
#model_name_or_path=luyaojie/uie-base-en
#model_name_or_path=google/flan-t5-xxl

# for TASK in re ner eet eea 
for TASK_CONFIG in ner
do
    for DATASET_CONFIG in ${TASK2DATASETS[${TASK_CONFIG}]}
    do
        CUDA_VISIBLE_DEVICES=0,1,2,3 python src/run_uie.py \
        --do_predict \
        --num_beams 1 \
        --repetition_penalty 1.0 \
        --predict_with_generate \
        --model_name_or_path ${model_name_or_path} \
        --data_dir ./data/ie_instruct \
        --task_config_dir ./configs/${TASK_CONFIG}_configs/${DATASET_CONFIG} \
        --instruction_file ./configs/instruction_config.json \
        --prompt_file ./prompts/instructUIE.json \
        --instruction_strategy multiple \
        --min_negative_labels -1 \
        --min_positive_labels -1 \
        --output_dir /storage_fast/zkhu/UIE-pp/output/ner/plo_all/uie_base_fft/checkpoint-2664/ \
        --input_record_file uie.record \
        --per_device_train_batch_size 24 \
        --per_device_eval_batch_size 96 \
        --gradient_accumulation_steps 10 \
        --learning_rate 5e-05 \
        --num_train_epochs 20 \
        --run_name ${model_name_or_path}-${TASK_CONFIG}-${DATASET_CONFIG} \
        --max_source_length 512 \
        --max_target_length 192 \
        --generation_max_length 192 \
        --max_num_instances_per_task 6000 \
        --max_num_instances_per_eval_task 6000 \
        --add_task_name False \
        --add_dataset_name False \
        --num_examples 0 \
        --overwrite_cache \
        --lr_scheduler_type constant \
        --warmup_steps 0 \
        --logging_strategy steps \
        --logging_steps 10 \
        --cache_dir ./huggingface \
        --ddp_find_unused_parameters False \
        --save_total_limit 3 \
        --over_sampling False \
        --load_best_model_at_end False \
        --metric_for_best_model eval_f1 \
        --only_save_best_model True \
        --predict_each_dataset_with_best False \
        --test_with_eval False \
        --use_test_as_eval False \
        --no_saving True \
        --evaluation_strategy epoch \
        --save_strategy epoch \
        #--resume_from_checkpoint /storage_fast/zkhu/UIE-pp/output/ner/plo_all/uie_base_fft/checkpoint-2664 \
        #--group_by_length \
        #--save_strategy steps \
        #--save_steps 1000 \
        #--evaluation_strategy steps \
        #--eval_steps 1000 \
        #--overwrite_output_dir \
    done
done