# remove files from checkpoint directory
# indicate the checkpoint directory
checkpoint_dir=./output/ADE_SciERC/flan-t5-xl_256bs
# remove files
rm -rf ${checkpoint_dir}/config.json
rm -rf ${checkpoint_dir}/generation_config.json
rm -rf ${checkpoint_dir}/optimizer.pt
rm -rf ${checkpoint_dir}/pytorch_model-00001-of-00002.bin
rm -rf ${checkpoint_dir}/pytorch_model-00002-of-00002.bin
rm -rf ${checkpoint_dir}/pytorch_model.bin.index.json
rm -rf ${checkpoint_dir}/rng_state.pth
rm -rf ${checkpoint_dir}/scheduler.pt
rm -rf ${checkpoint_dir}/special_tokens_map.json
rm -rf ${checkpoint_dir}/spiece.model
rm -rf ${checkpoint_dir}/tokenizer_config.json
rm -rf ${checkpoint_dir}/tokenizer.json
rm -rf ${checkpoint_dir}/trainer_state.json
rm -rf ${checkpoint_dir}/training_args.bin