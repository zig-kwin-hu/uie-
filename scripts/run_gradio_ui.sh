python src/generate.py \
    --model_type t5 \
    --base_model ./output/NER_other/flan-t5-base_256bs \
    --prompt_file ./prompts/instructUIE.json \
    --lora_weights None \