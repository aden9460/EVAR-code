#!/bin/bash
# Basic VAR Pruning Test Script

echo "======================================"
echo "VAR Basic Pruning Test"
echo "======================================"

cd /home/project/real_prune/Edgevar

# Ensure directories exist
# mkdir -p pruned_models

# Basic pruning with 20% sparsity
# CUDA_VISIBLE_DEVICES=0 python model_Edging_basic_v1.py \
#     --model_depth 16 \
#     --num_samples 1000 \
#     --sparsity 0.2 \
#     --minlayer 0 \
#     --maxlayer 16 \
#     --percdamp 0.01 \
#     --save_dir ./pruned_models \
#     --model_name qscaluefix_var_d16_0.2_1000sample_mag.pth \
#     --seed 0 \
#     --use_images \
#     --imagenet_dir "/home/project/ImageNet-1K" \
#     --prune_method magnitude


python model_Edging_basic_v1.py \
    --model_depth 16 \
    --sparsity 0.4 \
    --prune_method taylor \
    --taylor_type param_second \
    --num_taylor_samples 100 \
    --num_samples 1000 \
    --save_dir ./pruned_models \
    --model_name var_d16_0.4_llm-pruner.pth \
    --seed 0 \
    --use_images \
    --imagenet_dir "/home/project/ImageNet-1K" \
    # --use_selected_heads \
    # --selected_heads_json head_sharpness_images/pruning_plan_40pct.json
    # --non_uniform \
    # --non_uniform_strategy "log_increase" \
