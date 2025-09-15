#!/bin/bash

# Set environment variables if needed
# export HF_DATASETS_CACHE='/path/to/dataset_cache'
# export TRANSFORMERS_CACHE='/path/to/model_cache'
# export TRANSFORMERS_OFFLINE=1

# Default seed
seed=2021

# Run on a single GPU
for idrandom in 0;
do
  for ft_task in $(seq 0 0);
    do
      # For BERT-based models (ASC task)
      echo "Running BERT-based model on single GPU for ASC task"
      CUDA_VISIBLE_DEVICES=0 python finetune.py \
      --ft_task ${ft_task} \
      --idrandom ${idrandom} \
      --baseline 'adapter_ctr_asc_bert' \
      --seed ${seed} \
      --sequence_file 'asc' \
      --base_model_name_or_path 'bert-base-uncased' \
      --use_predefine_args
      
      # For BART-based models (Summarization task)
      echo "Running BART-based model on single GPU for Summarization task"
      CUDA_VISIBLE_DEVICES=0 python finetune.py \
      --ft_task ${ft_task} \
      --idrandom ${idrandom} \
      --baseline 'adapter_bcl_sum_bart' \
      --seed ${seed} \
      --sequence_file 'sum' \
      --base_model_name_or_path facebook/bart-base \
      --use_predefine_args
    done
done 