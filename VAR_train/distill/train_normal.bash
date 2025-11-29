
cd /home/project/real_prune/VAR_train

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun \
  --nnodes=1 \
  --nproc_per_node=7 \
  --node_rank=0 \
  distill/train_distill.py \
  --enable_distillation=1 \
  --teacher_model_path="/home/project/daily/AR/model_zoo/var_d16.pth" \
  --teacher_depth=16 \
  --distill_type="normal" \
  --distill_alpha=0.5 \
  --distill_beta=0.5 \
  --depth=16 --bs=256 --ep=20 --fp16=1 --alng=1e-3 --wpe=0.1 --sparsity=0.4 \
  --var_path="/home/project/real_prune/VAR_train/train_result/distill_d16_normal_256/ar-ckpt-best.pth" \
  --vae_path="/home/project/daily/AR/model_zoo/vae_ch160v4096z32.pth" \
  --data_path="/home/project/ImageNet-1K" \
  --local_out_dir_path="/home/project/real_prune/VAR_train/train_result/fix_distill_d16_normal_256_from20_progresswithoutsaware" \
   --pg=0.6 \
   --pg0=3