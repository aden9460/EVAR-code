

#!/bin/bash


PYTHON_SCRIPT="/path/to/your_script.py"
set -e
while true; do

    MEM1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7)

    if [ "$MEM1" -lt 10 ]; then

        sleep 3


        MEM2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7)

        if [ "$MEM2" -lt 10 ]; then

            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun  \
            --nnodes=1 \
            --nproc_per_node=8 \
            --node_rank=0 \
            train.py \
            --depth=16 --bs=320 --ep=1 --fp16=1 --alng=1e-3 --wpe=0.1 --sparsity=0.4 --local_out_dir_path="/home/project/real_prune/VAR_train/qscaluefix_var_d16_0.4_mag_1epoch" --data_path="/home/project/ImageNet-1K" \
            --var_path="/home/project/real_prune/slimvar/pruned_models/qscaluefix_var_d16_0.4_1000sample_mag.pth" \
            --vae_path='/home/project/daily/AR/model_zoo/vae_ch160v4096z32.pth'


            break
        else

        fi
    else

    fi


    sleep 500
done


# torchrun \
#   --nproc_per_node=6 \
#   --nnodes=2 \
#   --node_rank=1 \
#  train.py \
#   --depth=24 --bs=252 --ep=10 --fp16=1 --alng=1e-3 --wpe=0.1 --sparsity=0.2 --local_out_dir_path="/wanghuan/data/wangzefang/slim_VAR_copy/VAR/d20_0.2_0-10" --data_path="/wanghuan/data/wangzefang/ImageNet-1K/"
