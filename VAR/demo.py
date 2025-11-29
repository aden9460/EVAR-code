################## 1. Download checkpoints and build models
import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models import VQVAE, build_vae_var
import time
MODEL_DEPTH = 16    # TODO: =====> please specify MODEL_DEPTH <=====
assert MODEL_DEPTH in {16, 20, 24, 30}


# download checkpoint
hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
vae_ckpt, var_ckpt = '/home/wangzefang/Project/AR/model_zoo/vae_ch160v4096z32.pth', f'/home/wangzefang/Project/AR/model_zoo/var_d{MODEL_DEPTH}.pth'
if not osp.exists(vae_ckpt): os.system(f'wget {hf_home}/{vae_ckpt}')
if not osp.exists(var_ckpt): os.system(f'wget {hf_home}/{var_ckpt}')

# build vae, var
patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if 'vae' not in globals() or 'var' not in globals():
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
    )

# load checkpoints
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
vae.eval(), var.eval()
for p in vae.parameters(): p.requires_grad_(False)
for p in var.parameters(): p.requires_grad_(False)
print(f'prepare finished.')

############################# 2. Sample with classifier-free guidance

# set args
seed = 0 #@param {type:"number"}
torch.manual_seed(seed)
num_sampling_steps = 250 #@param {type:"slider", min:0, max:1000, step:1}
cfg = 4 #@param {type:"slider", min:1, max:10, step:0.1}
class_labels = (8,8,5)  #@param {type:"raw"}
class_labels = torch.arange(1)
more_smooth = False # True for more smooth output

# seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# run faster
tf32 = True
torch.backends.cudnn.allow_tf32 = bool(tf32)
torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
torch.set_float32_matmul_precision('high' if tf32 else 'highest')

# sample
B = len(class_labels)
label_B: torch.LongTensor = torch.tensor(class_labels, device=device)
# 添加延迟统计
total_times = []
total_images = []

# 运行10次并收集数据
for i in range(10):
    with torch.inference_mode():
        start = time.time()
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
            recon_B3HW = var.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.95, g_seed=seed, more_smooth=more_smooth)
        end = time.time()
        elapsed = end - start
        total_times.append(elapsed)
        total_images.append(B)
        print(f'运行 #{i+1}: 生成 {B} 张图片, 用时 {elapsed:.2f} 秒, 速度 {B/elapsed:.2f} img/s')

# 计算并打印统计信息
avg_time = sum(total_times) / len(total_times)
avg_speed = sum(total_images) / sum(total_times)
min_time = min(total_times)
max_time = max(total_times)

print('\n延迟统计:')
print(f'平均延迟: {avg_time:.2f} 秒')
print(f'最短延迟: {min_time:.2f} 秒')
print(f'最长延迟: {max_time:.2f} 秒')
print(f'平均速度: {avg_speed:.2f} img/s')

# 保存最后一次生成的图像
chw = torchvision.utils.make_grid(recon_B3HW, nrow=8, padding=0, pad_value=1.0)
chw = chw.permute(1, 2, 0).mul(255).cpu().numpy()
chw = PImage.fromarray(chw.astype(np.uint8))
img_path = f"./ori.png"
chw.save(img_path)
chw.show()