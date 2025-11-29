
set -e
while true; do
    # 第一次检测 GPU7 显存占用（MiB）
    MEM1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2)

    if [ "$MEM1" -lt 10 ]; then
        echo "GPU7 第一次检测显存占用为 0 MiB，3 秒后再次确认..."
        sleep 3

        # 第二次检测
        MEM2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2)

        if [ "$MEM2" -lt 10 ]; then
            echo "GPU7 连续两次显存占用为 0 MiB，执行 Python 程序..."
            cd /home/project/real_prune/VAR_train

            CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun \
              --nnodes=1 \
              --nproc_per_node=7 \
              --node_rank=0 \
              distill/train_distill.py \
              --enable_distillation=1 \
              --teacher_model_path="/home/project/daily/AR/model_zoo/var_d16.pth" \
              --teacher_depth=16 \
              --distill_type="scale_aware" \
              --scale_weights_str="2.0,1.8,1.6,1.4,1.2,1.0,0.8,0.6,0.4,0.2" \
              --distill_alpha=0.7 \
              --distill_beta=0.3 \
              --depth=16 --bs=256 --ep=20 --fp16=1 --alng=1e-3 --wpe=0.1 --sparsity=0.2 \
              --var_path="/home/project/real_prune/slimvar/pruned_models/qscaluefix_var_d16_0.2_1000sample_basic.pth" \
              --vae_path="/home/project/daily/AR/model_zoo/vae_ch160v4096z32.pth" \
              --data_path="/home/project/ImageNet-1K" \
              --local_out_dir_path="/home/project/real_prune/VAR_train/distill_d16_scale_aware_256_0.6_3_from0_20epoch_20%" \
              --pg=0.6 \
              --pg0=1

            break
        else
            echo "GPU7 第二次检测占用为 ${MEM2} MiB，继续等待..."
        fi
    else
        echo "GPU7 第一次检测占用为 ${MEM1} MiB，等待中..."
    fi

    # 每 10 秒检测一次
    sleep 500
done
