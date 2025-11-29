#!/usr/bin/env python3
"""
VAR Model Basic Pruning Script

This script implements the basic pruning pipeline without advanced features:
- Full-scale activation collection (all 680 tokens together)
- OBS (Optimal Brain Surgeon) Hessian-based importance evaluation
- Torch-Pruning structured pruning

Advanced features (controlled by parameters) will be added separately.
"""


import time
import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from transformers import set_seed
import os.path as osp
import torch_pruning as tp
import sys
sys.path.append("VAR/")
# Disable default parameter initialization for faster speed
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    """Recursively find all layers of specified types"""
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def get_module_by_name(layer, name):
    """Get module by name path (e.g., 'attn.proj')"""
    module = layer
    for attr in name.split('.'):
        module = getattr(module, attr)
    return module


class VARCatcher(nn.Module):
    """Catcher class to capture VAR block inputs for calibration"""
    def __init__(self, num_samples, seqlen, hidden_size, cache_dev='cuda', dtype=torch.float32):
        super().__init__()
        self.layer_inputs = torch.zeros(
            (num_samples, seqlen, hidden_size),
            dtype=dtype, device=cache_dev
        )
        self.row_idx = 0
        self.cache_dev = cache_dev

    def forward(self, x, cond_BD=None, attn_bias=None):
        """Capture input and interrupt forward pass"""
        batch_size = x.shape[0]

        # Handle batch input - store each sample in the batch
        if self.row_idx + batch_size <= self.layer_inputs.shape[0]:
            self.layer_inputs[self.row_idx:self.row_idx + batch_size] = x.detach()
            self.row_idx += batch_size
        else:
            # Handle last batch that might be smaller
            remaining = self.layer_inputs.shape[0] - self.row_idx
            if remaining > 0:
                self.layer_inputs[self.row_idx:self.row_idx + remaining] = x[:remaining].detach()
                self.row_idx += remaining

        raise ValueError("VARCatcher: Activation captured successfully")


def check_sparsity(model):
    """Check the sparsity of each layer and overall model"""
    layers = model.blocks
    count = 0
    total_params = 0

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W == 0).sum().item()
            total_params += W.numel()
            sub_count += (W == 0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    return float(count) / total_params


def load_var_model(model_depth, vae_ckpt_path, var_ckpt_path, device='cuda'):
    """
    Load VAR and VQVAE models

    Args:
        model_depth: VAR model depth (16, 20, 24, or 30)
        vae_ckpt_path: path to VQVAE checkpoint
        var_ckpt_path: path to VAR checkpoint
        device: device to load model on

    Returns:
        vae, var: loaded models
    """
    from VAR.models import build_vae_var

    assert model_depth in {16, 20, 24, 30}, f"Invalid model_depth: {model_depth}"

    if not osp.exists(vae_ckpt_path):
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_ckpt_path}")
    if not osp.exists(var_ckpt_path):
        raise FileNotFoundError(f"VAR checkpoint not found: {var_ckpt_path}")

    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

    # Build models
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=model_depth, shared_aln=False
    )

    # Load weights
    vae.load_state_dict(torch.load(vae_ckpt_path, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(var_ckpt_path, map_location='cpu'), strict=False)

    # Set to eval mode
    vae.eval()
    var.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in var.parameters():
        p.requires_grad_(False)

    print(f'VAR-d{model_depth} model loaded successfully.')
    return vae, var


@torch.no_grad()
def prepare_calibration_data(vae, num_samples, use_images=False, image_dir=None, final_reso=256):
    """
    Prepare calibration data: Get all tokens at once using VAR's original build_dataset

    Args:
        vae: VQVAE model
        num_samples: number of calibration samples
        use_images: if True, load real ImageNet images; if False, use class labels
        image_dir: path to ImageNet root directory (should contain 'train' subdirectory)
        final_reso: final image resolution (default: 256)

    Returns:
        calibration_labels: (num_samples,) class labels
        calibration_tokens: (num_samples, 679, 32) pre-encoded tokens, or None if use_images=False
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if use_images:
        # Use VAR's original build_dataset function
        print("Loading and encoding real ImageNet images using VAR's original build_dataset...")

        # Import VAR's original data loading function
        sys.path.insert(0, 'VAR')
        from VAR.utils.data import build_dataset

        # Load data using VAR's original method
        num_classes, train_set, val_set = build_dataset(
            data_path=image_dir,
            final_reso=final_reso,
            hflip=False,
            mid_reso=1.125
        )

        # Use validation set for calibration (consistent with training data augmentation)
        dataset = val_set

        # Balanced sampling by class: sample evenly from all classes
        print(f"  Balanced sampling: sampling {num_samples} samples from {num_classes} classes")

        # Simplified sampling: sample from first num_samples (ImageNet val set is usually ordered by class)
        # If num_samples >= num_classes, take at least 1 from each class
        if num_samples <= len(dataset):
            # Calculate step size to cover all classes
            step = len(dataset) // num_samples
            indices = torch.arange(0, len(dataset), step)[:num_samples]
        else:
            # If requested samples exceed dataset size, use all
            indices = torch.arange(len(dataset))

        print(f"  Actual sampling: {len(indices)} samples (uniformly spaced)")

        calibration_labels = []
        calibration_tokens = []

        batch_size = 8
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:min(i+batch_size, num_samples)]
            images = []
            labels = []

            for idx in batch_indices:
                img, label = dataset[int(idx)]
                images.append(img)
                labels.append(label)

            images = torch.stack(images).to(device)  # (B, 3, 256, 256), range [-1, 1]
            labels = torch.tensor(labels).to(device)

            # VQVAE encoding: img -> tokens (following trainer.py)
            gt_idx_Bl = vae.img_to_idxBl(images)  # List of 10 tensors
            x_BLCv = vae.quantize.idxBl_to_var_input(gt_idx_Bl)  # (B, 679, 32)

            calibration_labels.append(labels)
            calibration_tokens.append(x_BLCv.cpu())

            if (i + batch_size) % 64 == 0:
                print(f"  Processed {i + batch_size}/{num_samples} images")

        calibration_labels = torch.cat(calibration_labels, dim=0)
        calibration_tokens = torch.cat(calibration_tokens, dim=0)

        print(f"✓ Done! Obtained tokens for {num_samples} samples")
        return calibration_labels, calibration_tokens

    else:
        # Option 2: Simplified approach - use class labels (VAR handles internally)
        print("Using class labels as calibration data (simplified approach)")
        calibration_labels = torch.arange(0, num_samples).cuda()
        return calibration_labels, None


# @torch.no_grad() - Removed to support Taylor pruning with gradients
def model_Edging(model, calibration_labels, calibration_tokens, args):
    """
    Execute basic VAR model pruning with streaming activation propagation

    Args:
        model: VAR model
        calibration_labels: (num_samples,) class labels
        calibration_tokens: (num_samples, 679, 32) pre-encoded tokens, or None for label-only mode
        args: pruning arguments

    Returns:
        model: pruned model
    """
    from OBS import OBS

    device = "cuda" if torch.cuda.is_available() else 'cpu'
    layers = model.blocks
    num_samples = len(calibration_labels)

    print("\n" + "="*60)
    print("Starting VAR Model Pruning with Streaming Activations")
    print("="*60)
    print(f"Model depth: {len(layers)}")
    print(f"Pruning layers: {args.minlayer} to {args.maxlayer}")
    print(f"Target sparsity: {args.sparsity}")
    print(f"Calibration samples: {num_samples}")
    print(f"Using pre-encoded tokens: {calibration_tokens is not None}")
    print("="*60 + "\n")

    t_start = time.time()

    # Phase 1: Collect initial representations using VARCatcher
    print("Phase 1: Collecting initial layer inputs...")
    original_first_block = layers[0]
    var_catcher = VARCatcher(num_samples, 680, model.C, cache_dev=device)
    layers[0] = var_catcher

    batch_size = 16
    sample_count = 0
    for batch_idx in range(0, num_samples, batch_size):
        end_idx = min(batch_idx + batch_size, num_samples)
        batch_labels = calibration_labels[batch_idx:end_idx]

        try:
            if calibration_tokens is not None:
                # Use pre-encoded tokens (teacher forcing mode)
                batch_tokens = calibration_tokens[batch_idx:end_idx].to(device)
                model(batch_labels, batch_tokens)
            else:
                # Fallback to label-only mode - create dummy tokens
                for b in model.blocks[1:]:  # Skip the catcher
                    b.attn.kv_caching(True)
                # Create minimal dummy tokens for initialization (VAR requires both label and tokens)
                dummy_tokens = torch.zeros(len(batch_labels), 679, model.vae_proxy[0].Cvae, device=device)
                model(batch_labels, dummy_tokens)
                for b in model.blocks[1:]:
                    b.attn.kv_caching(False)
        except ValueError:
            # Expected interruption from VARCatcher
            pass

        sample_count += len(batch_labels)
        if batch_idx % 64 == 0:
            print(f"  Captured {sample_count}/{num_samples} samples")

    # Get the captured layer inputs and restore the original first block
    layer_inputs = var_catcher.layer_inputs.clone()  # (num_samples, 680, C)
    layers[0] = original_first_block
    del var_catcher
    torch.cuda.empty_cache()

    print(f"✓ Captured layer inputs shape: {layer_inputs.shape}")  # Should be (num_samples, 680, C)
    print(f"Phase 2: Layer-by-layer pruning and activation propagation...")

    # Phase 2: Layer-by-layer pruning with activation propagation
    for i in range(len(layers)):
        print(f"\n{'='*60}")
        print(f"Processing Layer {i}/{len(layers)-1}")
        print(f"{'='*60}")

        layer = layers[i].to(device)

        if args.minlayer <= i < args.maxlayer:
            all_module_dict = find_layers(layer)

            # Different collection strategies for different pruning methods
            sequential = [
                ["attn.proj", "ffn.fc2"],  # Collect both modules together
            ]

            for names in sequential:
                module_dict = {name: all_module_dict[name] for name in names}
                pruner_dict = {}

                # Step 1: Initialize pruners for ALL modules
                print(f"  Initializing pruners for: {names}")
                for name in module_dict:
                    pruner_dict[name] = OBS(module_dict[name], i, args)

                if args.prune_method in ["OBS", "magnitude"]:
                    # ========== OBS/Magnitude method: collect activation statistics ==========
                    print(f"  Collecting activations for {args.prune_method}...")

                    # Step 2: Register hooks on all modules
                    def add_batch(name):
                        def func(_, inp, out):
                            pruner_dict[name].add_batch(inp[0].data, out.data)
                        return func

                    handles = []
                    for name in module_dict:
                        handles.append(module_dict[name].register_forward_hook(add_batch(name)))

                    # Step 3: Collect activations using current layer_inputs
                    batch_size = 16
                    for batch_idx in range(0, num_samples, batch_size):
                        end_idx = min(batch_idx + batch_size, num_samples)

                        for j in range(batch_idx, end_idx):
                            layer_input = layer_inputs[j:j+1]  # (1, 680, C)
                            class_label = calibration_labels[j:j+1]
                            cond_BD = model.class_emb(class_label)  # (1, C)
                            cond_BD_or_gss = model.shared_ada_lin(cond_BD)
                            seq_len = layer_input.shape[1]
                            attn_bias = model.attn_bias_for_masking[:, :, :seq_len, :seq_len]

                            # Forward only (no gradients needed)
                            with torch.no_grad() if args.prune_method == "magnitude" else torch.enable_grad():
                                out = layer(x=layer_input, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)

                        if batch_idx % 64 == 0:
                            print(f"    Processing samples {batch_idx}/{num_samples}")

                    # Step 4: Remove hooks
                    for h in handles:
                        h.remove()

                elif args.prune_method == "taylor":
                    # ========== LLM-Pruner Taylor method: collect gradients ==========
                    print(f"  Using Taylor method: {getattr(args, 'taylor_type', 'param_mix')}")
                    taylor_type = getattr(args, 'taylor_type', 'param_mix')
                    num_taylor_samples = getattr(args, 'num_taylor_samples', min(20, num_samples))

                    # Phase 1: Collect second-order gradients sample-by-sample (if needed)
                    if taylor_type in ['param_second', 'param_mix']:
                        print(f"  Collecting second-order gradients (Hessian diagonal)...")

                        for j in range(num_taylor_samples):
                            layer_input = layer_inputs[j:j+1]
                            class_label = calibration_labels[j:j+1]
                            cond_BD = model.class_emb(class_label)
                            cond_BD_or_gss = model.shared_ada_lin(cond_BD)
                            seq_len = layer_input.shape[1]
                            attn_bias = model.attn_bias_for_masking[:, :, :seq_len, :seq_len]

                            # Forward
                            out = layer(x=layer_input, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)

                            # Construct loss (simple output norm)
                            loss = (out ** 2).sum()
                            loss.backward()

                            # Accumulate second-order gradients (Fixed: before zero_grad)
                            for name in pruner_dict:
                                pruner_dict[name].accumulate_hessian_diag()

                            model.zero_grad()

                            if (j + 1) % 10 == 0:
                                print(f"    Processed {j+1}/{num_taylor_samples} samples for Hessian")

                        # Normalize second-order gradients
                        for name in pruner_dict:
                            pruner_dict[name].finalize_hessian_diag()

                        print(f"  ✓ Second-order gradient collection completed")

                    # Phase 2: Collect first-order gradients (if needed)
                    if taylor_type in ['param_first', 'param_mix']:
                        print(f"  Collecting first-order gradients...")

                        # Use average loss from multiple samples
                        total_loss = 0
                        for j in range(num_taylor_samples):
                            layer_input = layer_inputs[j:j+1]
                            class_label = calibration_labels[j:j+1]
                            cond_BD = model.class_emb(class_label)
                            cond_BD_or_gss = model.shared_ada_lin(cond_BD)
                            seq_len = layer_input.shape[1]
                            attn_bias = model.attn_bias_for_masking[:, :, :seq_len, :seq_len]

                            out = layer(x=layer_input, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
                            total_loss += (out ** 2).sum()

                        total_loss /= num_taylor_samples
                        total_loss.backward()

                        # Capture first-order gradients
                        for name in pruner_dict:
                            pruner_dict[name].capture_first_order_grad()

                        print(f"  ✓ Gradient collection completed")

                else:
                    raise ValueError(f"Unknown pruning method: {args.prune_method}")

                # Step 5: Now prune each module in dependency order (attn.proj first, then ffn.fc2)
                prune_order = ["attn.proj", "ffn.fc2"]
                for name in prune_order:
                    sparsity = args.sparsity[i] if isinstance(args.sparsity, list) else args.sparsity
                    print(f"  Layer {i}: {name} - pruning {sparsity*100:.1f}% using {args.prune_method}")

                    # Select pruning method
                    if args.prune_method == "OBS":
                        idx = pruner_dict[name].struct_prune(
                            sparsity=sparsity,
                            percdamp=args.percdamp,
                            headsize=64 if name == "attn.proj" else 1,
                            layer_idx=i,
                        )
                    elif args.prune_method == "magnitude":
                        idx = pruner_dict[name].magnitude_prune(
                            sparsity=sparsity,
                            percdamp=args.percdamp,
                            headsize=64 if name == "attn.proj" else 1,
                            layer_idx=i,
                        )
                    elif args.prune_method == "taylor":
                        # Use LLM-Pruner Taylor method
                        idx = pruner_dict[name].taylor_prune_llm(
                            sparsity=sparsity,
                            percdamp=args.percdamp,
                            headsize=64 if name == "attn.proj" else 1,
                            layer_idx=i,
                            taylor_type=getattr(args, 'taylor_type', 'param_mix')
                        )
                    else:
                        raise ValueError(f"Unknown pruning method: {args.prune_method}")

                    pruner_dict[name].free()

                    # Execute Torch-Pruning
                    target_layer = get_module_by_name(model.blocks[i], name)

                    if name == "ffn.fc2":
                        target_layer_b = get_module_by_name(model.blocks[i], "ffn.fc1")
                        idx_list = idx.tolist()
                        tp.prune_linear_in_channels(target_layer, idx_list)
                        tp.prune_linear_out_channels(target_layer_b, idx_list)
                        print(f"    ✓ Pruned {len(idx_list)} channels from fc2 input and fc1 output")

                    elif name == "attn.proj":
                        # Fixed: use current layer's num_heads instead of global model.num_heads
                        current_num_heads = model.blocks[i].attn.num_heads
                        new_num_heads = torch.round(torch.tensor(current_num_heads * (1 - sparsity))).int()
                        model.blocks[i].attn.num_heads = new_num_heads

                        idx_m = idx.to(dtype=torch.long)
                        idx_list = idx.tolist()
                        keep_idxs = list(set(range(target_layer.in_features)) - set(idx_list))

                        # Debug: print key information
                        print(f"    ✓ Attention pruning details:")
                        print(f"      Current heads: {current_num_heads} -> New heads: {new_num_heads}")
                        print(f"      Pruned channels: {len(idx_list)}")
                        print(f"      Kept channels: {len(keep_idxs)}")

                        # Update biases
                        model.blocks[i].attn.q_bias = nn.Parameter(model.blocks[i].attn.q_bias.data[keep_idxs])
                        zero_k_bias = model.blocks[i].attn.zero_k_bias.data[keep_idxs]
                        model.blocks[i].attn.register_buffer('zero_k_bias', zero_k_bias)
                        model.blocks[i].attn.v_bias = nn.Parameter(model.blocks[i].attn.v_bias.data[keep_idxs])

                        # Update scale parameter - FIXED: Preserve original learned values
                        head_dim = 64  # VAR uses fixed head_dim=64

                        # Calculate which heads are removed (idx contains channel indices to remove)
                        removed_heads = set((idx_m // head_dim).tolist())
                        all_heads = set(range(current_num_heads))  # Fixed: use current_num_heads
                        keep_heads = sorted(list(all_heads - removed_heads))

                        # Preserve the learned scale_mul values for remaining heads
                        old_scale_mul = model.blocks[i].attn.scale_mul_1H11.data  # (1, current_num_heads, 1, 1)
                        new_scale_mul = old_scale_mul[0, keep_heads, 0, 0].view(1, -1, 1, 1)

                        model.blocks[i].attn.scale_mul_1H11 = nn.Parameter(
                            new_scale_mul.clone().to(device),
                            requires_grad=True
                        )

                        # Debug info
                        print(f"    ✓ Preserved scale_mul for heads {keep_heads}")
                        print(f"      Removed heads: {sorted(removed_heads)}")
                        print(f"      scale_mul shape: {old_scale_mul.shape} -> {new_scale_mul.shape}")
                        print(f"      scale_mul range: [{new_scale_mul.exp().min().item():.3f}, {new_scale_mul.exp().max().item():.3f}]")

                        # Prune proj
                        tp.prune_linear_in_channels(target_layer, idx_list)

                        # Prune mat_qkv
                        target_layer_b = get_module_by_name(model.blocks[i], "attn.mat_qkv")
                        hidden = current_num_heads * 64  # Fixed: use current_num_heads

                        rm_feat_q = idx_m
                        rm_qkv = torch.cat([
                            rm_feat_q,
                            rm_feat_q + hidden,
                            rm_feat_q + 2*hidden
                        ], dim=0)

                        rm_qkv_list = torch.unique(rm_qkv.to("cpu")).sort().values.tolist()
                        tp.prune_linear_out_channels(target_layer_b, rm_qkv_list)

                        print(f"    ✓ Pruned {len(idx_list)} channels ({len(idx_list)//64} heads)")
                        print(f"    ✓ New head count: {model.blocks[i].attn.num_heads}")
                        print(f"    ✓ mat_qkv shape after pruning: {target_layer_b.weight.shape}")

                del pruner_dict
                torch.cuda.empty_cache()

            # Print layer shapes after pruning
            print(f"\n  Layer {i} shapes after pruning:")
            print(f"    ffn.fc1:      {model.blocks[i].ffn.fc1.weight.shape}")
            print(f"    ffn.fc2:      {model.blocks[i].ffn.fc2.weight.shape}")
            print(f"    attn.mat_qkv: {model.blocks[i].attn.mat_qkv.weight.shape}")
            print(f"    attn.proj:    {model.blocks[i].attn.proj.weight.shape}")

        # Update layer_inputs to the output of current layer (for next layer's input)
        print(f"  Updating layer_inputs for layer {i+1}...")

        # Process samples in batches to update layer_inputs
        batch_size = 16
        for batch_idx in range(0, num_samples, batch_size):
            end_idx = min(batch_idx + batch_size, num_samples)

            for j in range(batch_idx, end_idx):
                # Forward through the current layer to get output for next layer
                layer_input = layer_inputs[j:j+1]  # (1, 680, C)

                # Get class embedding for this sample
                class_label = calibration_labels[j:j+1]
                cond_BD = model.class_emb(class_label)  # (1, C)
                cond_BD_or_gss = model.shared_ada_lin(cond_BD)

                # Get attention bias - adapt to current layer's head count after pruning
                seq_len = layer_input.shape[1]
                current_num_heads = layer.attn.num_heads if hasattr(layer.attn, 'num_heads') else model.num_heads
                # Take only the needed number of heads from the original attention bias
                original_attn_bias = model.attn_bias_for_masking[:, :, :seq_len, :seq_len]
                if current_num_heads < model.num_heads:
                    # Layer has been pruned, use only the first current_num_heads
                    attn_bias = original_attn_bias[:, :current_num_heads, :, :]
                else:
                    attn_bias = original_attn_bias

                # Forward through the layer with proper VAR inputs
                with torch.no_grad():
                    layer_output = layer(x=layer_input, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
                    layer_inputs[j] = layer_output.squeeze(0)  # Update in-place: (680, C)

        # Move layer to CPU to save memory, keep GPU for next layer
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    t_end = time.time()
    print(f"\n{'='*60}")
    print(f"Pruning completed in {t_end - t_start:.2f}s")
    print(f"{'='*60}\n")

    return model


def main(args):
    print("\n" + "="*60)
    print("VAR Model Basic Pruning")
    print("="*60)

    # 1. Load model
    print("\n[1/5] Loading model...")
    vae_ckpt = args.vae_ckpt if args.vae_ckpt else f'./checkpoints/vae_ch160v4096z32.pth'
    var_ckpt = args.var_ckpt if args.var_ckpt else f'./checkpoints/var_d{args.model_depth}.pth'

    vae, var = load_var_model(args.model_depth, vae_ckpt, var_ckpt, device='cuda')

    # 2. Set pruning layer range
    args.minlayer = max(args.minlayer, 0)
    args.maxlayer = min(args.maxlayer, args.model_depth)

    # 3. Configure non-uniform sparsity if needed
    if args.non_uniform:
        print("\n[2/5] Configuring non-uniform pruning...")
        assert 0 <= args.min_sparsity <= args.max_sparsity < 1

        if args.non_uniform_strategy in ('log_increase', 'log_decrease'):
            linear_space = np.arange(0, args.maxlayer - args.minlayer)
            args.sparsity = args.min_sparsity + \
                (args.max_sparsity - args.min_sparsity) / np.log(32) * np.log(1 + linear_space)
            args.sparsity = [0] * args.minlayer + list(args.sparsity)
            if args.non_uniform_strategy == 'log_decrease':
                args.sparsity = args.sparsity[::-1]

        print(f"  Sparsity schedule: {[f'{s:.3f}' for s in args.sparsity[args.minlayer:args.maxlayer]]}")
    else:
        print(f"\n[2/5] Using uniform sparsity: {args.sparsity}")

    # 4. Prepare calibration data
    print("\n[3/5] Preparing calibration data...")
    if hasattr(args, 'use_images') and args.use_images:
        # Use real ImageNet images, encode all tokens at once
        calibration_labels, calibration_tokens = prepare_calibration_data(
            vae, args.num_samples,
            use_images=True,
            image_dir=args.imagenet_dir
        )
    else:
        # Simplified approach: use class labels
        calibration_labels, calibration_tokens = prepare_calibration_data(
            vae, args.num_samples,
            use_images=False
        )
    print(f"  Calibration samples: {len(calibration_labels)}")

    # 5. Print model statistics
    state_dict = var.state_dict()
    total_params = sum(v.numel() for v in state_dict.values()) / 1e9
    print(f"  Total parameters: {total_params:.2f}B")

    # Enable gradients for Taylor pruning
    if args.prune_method == "taylor":
        print("\n  Enabling gradients for Taylor pruning...")
        var.requires_grad_(True)

    # 6. Execute pruning
    if isinstance(args.sparsity, list) or args.sparsity > 0:
        print("\n[4/5] Executing pruning...")
        tick = time.time()
        var = model_Edging(var, calibration_labels, calibration_tokens, args)
        tock = time.time()
        print(f"Total pruning time: {tock - tick:.2f}s")

    # 7. Check sparsity
    print("\n[5/5] Checking sparsity...")
    print("="*60)
    sparsity_ratio = check_sparsity(var)
    print(f"Overall sparsity: {sparsity_ratio:.4f}")
    print("="*60)

    # 8. Print pruned model statistics
    state_dict = var.state_dict()
    pruned_params = sum(v.numel() for v in state_dict.values()) / 1e9
    print(f"\nParameters after pruning: {pruned_params:.2f}B")
    print(f"Parameter reduction: {(1 - pruned_params/total_params)*100:.2f}%")

    # # 9. Measure inference speed
    # print("\nMeasuring inference speed...")
    # example_input = torch.tensor([0]).cuda()
    # for b in var.blocks:
    #     b.attn.kv_caching(True)

    # # Warmup
    # for _ in range(3):
    #     _ = var(example_input)

    # torch.cuda.synchronize()
    # start = time.time()
    # for _ in range(10):
    #     _ = var(example_input)
    # torch.cuda.synchronize()
    # end = time.time()

    # avg_time = (end - start) / 10 * 1000  # ms
    # print(f"Average inference time: {avg_time:.2f} ms")

    # # 10. Count MACs and parameters
    # macs, nparams = tp.utils.count_ops_and_params(var, example_input, layer_wise=False)
    # print(f"\nFinal statistics:")
    # print(f"  Parameters: {nparams / 1e6:.2f}M")
    # print(f"  MACs: {macs / 1e9:.2f}G")

    # 11. Save model
    if args.save_dir:
        print(f"\nSaving model...")
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, args.model_name)
        torch.save(var.state_dict(), save_path)
        print(f"Model saved to: {save_path}")

    print("\n" + "="*60)
    print("Pruning complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAR Model Basic Pruning")

    # Model configuration
    parser.add_argument(
        "--model_depth", type=int, default=16,
        choices=[16, 20, 24, 30],
        help="VAR model depth"
    )
    parser.add_argument(
        "--vae_ckpt", type=str, default="",
        help="Path to VQVAE checkpoint (default: ./checkpoints/vae_ch160v4096z32.pth)"
    )
    parser.add_argument(
        "--var_ckpt", type=str, default="",
        help="Path to VAR checkpoint (default: ./checkpoints/var_d{depth}.pth)"
    )

    # Calibration data
    parser.add_argument(
        "--num_samples", type=int, default=256,
        help="Number of calibration samples"
    )
    parser.add_argument(
        "--use_images", action="store_true",
        help="Use real ImageNet images for calibration (pre-encode tokens). If False, use class labels."
    )
    parser.add_argument(
        "--imagenet_dir", type=str, default="./data/imagenet",
        help="Path to ImageNet training directory (required if --use_images is set)"
    )

    # Pruning configuration
    parser.add_argument(
        "--sparsity", type=float, default=0.2,
        help="Target pruning ratio (0-1)"
    )
    parser.add_argument(
        "--minlayer", type=int, default=0,
        help="Start pruning from this layer"
    )
    parser.add_argument(
        "--maxlayer", type=int, default=16,
        help="Prune up to this layer (exclusive)"
    )
    parser.add_argument(
        "--percdamp", type=float, default=0.01,
        help="Percent of average Hessian diagonal for dampening"
    )

    # Non-uniform pruning
    parser.add_argument(
        "--non_uniform", action="store_true",
        help="Use non-uniform pruning strategy"
    )
    parser.add_argument(
        "--non_uniform_strategy", type=str, default='log_increase',
        choices=["log_increase", "log_decrease"],
        help="Non-uniform pruning strategy"
    )
    parser.add_argument(
        "--min_sparsity", type=float, default=0,
        help="Minimum sparsity for non-uniform pruning"
    )
    parser.add_argument(
        "--max_sparsity", type=float, default=0.6,
        help="Maximum sparsity for non-uniform pruning"
    )

    # Other options
    parser.add_argument(
        "--prune_method", type=str, default="OBS",
        choices=["OBS", "magnitude", "taylor"],
        help="Pruning method: OBS (Optimal Brain Surgeon, Hessian-based), magnitude, or taylor"
    )
    parser.add_argument(
        "--taylor_type", type=str, default="param_mix",
        choices=["param_first", "param_second", "param_mix"],
        help="Taylor importance type: param_first (1st order), param_second (2nd order), param_mix (mixed, recommended). Only used when --prune_method=taylor"
    )
    parser.add_argument(
        "--num_taylor_samples", type=int, default=20,
        help="Number of samples for Taylor gradient collection. Only used when --prune_method=taylor. Recommended: 10-50"
    )
    parser.add_argument(
        "--no_compensate", action="store_true",
        help="Skip error compensation in OBS"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./pruned_models",
        help="Directory to save pruned model"
    )
    parser.add_argument(
        "--model_name", type=str, default="var_0.2_256sample_prune.pth",
        help="Name of saved model file"
    )

    args = parser.parse_args()

    print("\nArguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    set_seed(args.seed)
    main(args)
