
import torch
import os
from typing import List, Optional
import dist


def load_teacher_model(args):

    from models import build_vae_var

    print(f"[ {args.teacher_model_path}")
    print(f"[: {args.teacher_depth}")


    student_sparsity = args.sparsity


    args.sparsity = 0.0


    try:
        teacher_vae, teacher_var = build_vae_var(
            V=4096, Cvae=32, ch=160, share_quant_resi=4,     
            device=dist.get_device(), patch_nums=args.patch_nums,
            num_classes=1000,  
            depth=args.teacher_depth, 
            shared_aln=args.saln, attn_l2_norm=args.anorm,
            flash_if_available=args.fuse, fused_if_available=args.fuse,
            init_adaln=args.aln, init_adaln_gamma=args.alng,
            init_head=args.hd, init_std=args.ini, args=args
        )
    finally:

        args.sparsity = student_sparsity
        print(f" sparsity={student_sparsity}")

    print("")
    checkpoint = torch.load(args.teacher_model_path, map_location='cpu')

    if 'trainer' in checkpoint:

        if 'var_wo_ddp' in checkpoint['trainer']:
            teacher_var.load_state_dict(checkpoint['trainer']['var_wo_ddp'], strict=True)
            print("")
        else:
            raise KeyError("")

        teacher_var.load_state_dict(checkpoint, strict=True)
        print("")

    teacher_var.eval()
    frozen_params = 0
    for param in teacher_var.parameters():
        param.requires_grad = False
        frozen_params += param.numel()

    param_count = count_parameters(teacher_var)
    print(f": {param_count:.2f}M")
    print(f" {frozen_params/1e6:.2f}M")

    return teacher_var


def count_parameters(model) -> float:

    return sum(p.numel() for p in model.parameters()) / 1e6


def parse_scale_weights(scale_weights_str: str) -> List[float]:

    try:
        weights = [float(x.strip()) for x in scale_weights_str.split(',')]
        if len(weights) != 10:
            raise ValueError(f"{len(weights)}个")
        if any(w < 0 for w in weights):
            raise ValueError("")
        return weights
    except ValueError as e:
        raise ValueError(f": {e}")


def parse_feature_layers(feature_layers_str: str) -> List[int]:

    try:
        layers = [int(x.strip()) for x in feature_layers_str.split(',')]
        if any(layer < 0 for layer in layers):
            raise ValueError("")
        return layers
    except ValueError as e:
        raise ValueError(f": {e}")


def validate_teacher_student_compatibility(teacher_model, student_model, args):

    # 检查参数数量
    teacher_params = count_parameters(teacher_model)
    student_params = count_parameters(student_model)

    print(f"[ {teacher_params:.2f}M")
    print(f"[ {student_params:.2f}M")
    print(f"[ {student_params/teacher_params:.2%}")


def get_scale_boundaries():

    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)  # VAR的标准patch配置
    boundaries = [0]
    current_pos = 0

    for pn in patch_nums:
        current_pos += pn * pn
        boundaries.append(current_pos)

    return boundaries


def compute_scale_weights(strategy: str = 'linear_decay', **kwargs) -> List[float]:

    if strategy == 'linear_decay':
        # 线性衰减：从高到低
        start_weight = kwargs.get('start_weight', 2.0)
        end_weight = kwargs.get('end_weight', 0.2)
        step = (start_weight - end_weight) / 9
        return [start_weight - i * step for i in range(10)]

    elif strategy == 'exponential_decay':

        base = kwargs.get('base', 0.5)
        scale = kwargs.get('scale', 4.0)
        return [scale * (base ** i) for i in range(10)]

    elif strategy == 'uniform':

        weight = kwargs.get('weight', 1.0)
        return [weight] * 10

    elif strategy == 'early_focus':

        high_weight = kwargs.get('high_weight', 3.0)
        low_weight = kwargs.get('low_weight', 0.1)
        focus_scales = kwargs.get('focus_scales', 4)
        weights = []
        for i in range(10):
            if i < focus_scales:
                weights.append(high_weight)
            else:
                weights.append(low_weight)
        return weights

    else:
        raise ValueError(f": {strategy}")


def create_distill_config_summary(args) -> str:

    if not args.enable_distillation:
        return ""

    summary = []
    summary.append("")
    summary.append(f": {args.distill_type}")
    summary.append(f": {os.path.basename(args.teacher_model_path)} {args.teacher_depth})")
    summary.append(f"={args.distill_alpha:.2f}, ={args.distill_beta:.2f}")
    summary.append(f": {args.distill_temperature}")

    if args.distill_type in ['scale_aware', 'both']:
        weights = args.scale_weights
        summary.append(f" [{weights[0]:.1f}, {weights[1]:.1f}, ..., {weights[-1]:.1f}]")

    if args.use_feature_distill:
        summary.append(f"{args.feature_layers}")

    if args.use_attention_distill:
        summary.append("")

    summary.append("=" * 30)

    return "\n".join(summary)


def log_distillation_step(step: int, task_loss: float, distill_loss: float,
                         total_loss: float, logits_similarity: Optional[float] = None):

    log_msg = f"[Step {step}] task loss: {task_loss:.4f}, distill loss: {distill_loss:.4f}, total loss: {total_loss:.4f}"

    if logits_similarity is not None:
        log_msg += f", similarity: {logits_similarity:.4f}"

    print(log_msg)