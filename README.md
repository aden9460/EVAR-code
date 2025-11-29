# EVAR: Edge Visual Autoregressive Models via Principled Pruning

This repository implements structured pruning and knowledge distillation for Visual Autoregressive (VAR) models, enabling efficient image generation with reduced model size and computational cost.

## Overview

EVAR provides a complete framework for:
- **Structured Neural Network Pruning**: Reduce model parameters while maintaining performance
- **Knowledge Distillation**: Transfer knowledge from teacher models to pruned student models
- **OBS-based Importance Evaluation**: Use Optimal Brain Surgeon for intelligent pruning decisions

## Features

- Multi-scale VAR model implementation with VQVAE encoding
- Magnitude-based and Taylor expansion pruning methods
- Scale-aware distillation for multi-resolution image generation
- Support for distributed training with PyTorch DDP
- Integration with torch-pruning library for structured pruning

## Project Structure

```
Evar/
├── model_Edging_basic_v1.py      # Main pruning script
├── OBS.py                        # OBS (Optimal Brain Surgeon) Hessian computation
├── run_basic_pruning.bash        # Example pruning script
├── requirements.txt              # Python dependencies
│
├── VAR/                          # Original VAR model implementation
│   ├── models/                   # Model architectures (VAR, VQVAE, etc.)
│   ├── utils/                    # Training utilities
│   ├── train.py                  # Standard training script
│   └── trainer.py                # VARTrainer class
│
└── VAR_train/                    # Enhanced training with distillation
    ├── models/                   # Enhanced model implementations
    ├── utils/                    # Enhanced utilities
    ├── distill/                  # Knowledge distillation module
    │   ├── distillation_trainer.py
    │   ├── distillation_losses.py
    │   ├── distill_utils.py
    │   └── train_distill.py
    └── train_var.bash            # Training script example
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/aden9460/EVAR-code.git
cd EVAR-code
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install Flash Attention for faster training:
```bash
pip install flash-attn
```

## Quick Start

### 1. Model Pruning

Prune a pre-trained VAR model using magnitude or Taylor-based importance:

```bash
bash run_basic_pruning.bash
```

Or run directly with Python:
```bash
python model_Edging_basic_v1.py \
    --model_path /path/to/checkpoint.pth \
    --sparsity 0.3 \
    --prune_method taylor \
    --output_dir ./pruned_models
```

### 2. Knowledge Distillation Training

Train a pruned student model with knowledge distillation:

```bash
cd VAR_train/distill
bash train_aware.bash
```

Or with custom parameters:
```bash
python train_distill.py \
    --teacher_model /path/to/teacher.pth \
    --student_model /path/to/pruned_student.pth \
    --enable_distillation \
    --distill_type scale_aware \
    --distill_temperature 4.0
```

### 3. Standard Training

Train a VAR model from scratch:

```bash
cd VAR
python train.py \
    --data_path /path/to/imagenet \
    --depth 16 \
    --embed_dim 1024 \
    --num_heads 16 \
    --batch_size 256
```

## Key Parameters

### Pruning Parameters
- `--sparsity`: Target pruning ratio (e.g., 0.3 for 30% reduction)
- `--prune_method`: Choose from `magnitude` or `taylor`
- `--taylor_type`: For Taylor pruning: `param_first`, `param_second`, or `param_mix`
- `--percdamp`: Damping parameter for Hessian computation (default: 0.01)

### Distillation Parameters
- `--enable_distillation`: Enable knowledge distillation
- `--distill_type`: Choose from `normal`, `scale_aware`, or `both`
- `--distill_temperature`: Temperature for softmax distillation (default: 4.0)
- `--distill_alpha`: Weight for distillation loss
- `--distill_beta`: Weight for task loss

### Model Configuration
- `--depth`: Number of transformer layers (default: 16)
- `--embed_dim`: Embedding dimension (default: 1024)
- `--num_heads`: Number of attention heads (default: 16)
- `--patch_nums`: Multi-scale patch configurations (default: 1,2,3,4,5,6,8,10,13,16)

## Results

The pruning and distillation framework achieves:
- **1.8× speedup** on iPad Pro (M4): 494ms → 277ms for single-image inference
- **26% parameter reduction**: 310M → 230M parameters (40% sparsity)
- **Minimal quality loss**: FID increases from 3.55 to 3.91 (~10% relative degradation)
- Competitive performance with state-of-the-art AR models while enabling edge deployment

### Performance Comparison (ImageNet 256×256)

| Model | Parameters | Pruning | FID↓ | IS↑ | Precision↑ | Recall↑ |
|-------|-----------|---------|------|-----|-----------|---------|
| VAR-d16 | 310M | - | 3.55 | 274.4 | 0.84 | 0.51 |
| EVAR (20%) | 270M | 20% | 3.67 | 57.78 | 0.81 | 0.51 |
| EVAR (40%) | 230M | 40% | **3.91** | 57.23 | 0.81 | 0.51 |

### Edge Deployment Results

| Device | Model | Latency (ms) | Speedup |
|--------|-------|-------------|---------|
| iPad Pro (M4) | VAR-d16 | 494 | 1.0× |
| iPad Pro (M4) | EVAR (40%) | **277** | **1.8×** |

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA compatible GPU (recommended)
- 16GB+ GPU memory for training

See `requirements.txt` for complete dependency list.

## Acknowledgements

This project builds upon and references the following excellent works:

- [VAR](https://github.com/FoundationVision/VAR): Visual Autoregressive Modeling - Scalable Image Generation via Next-Scale Prediction
- [DepGraph](https://github.com/VainF/Torch-Pruning): Towards Any Structural Pruning
- [Optimal Brain Compression](https://arxiv.org/abs/2208.11580): A Framework for Accurate Post-Training Quantization and Pruning
- [SlimGPT](https://arxiv.org/abs/2401.02547): Layer-wise Structured Pruning for Large Language Models

We extend our gratitude to the authors for their outstanding contributions to the community.

## License

This project is released under the MIT License. See the LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact the repository maintainer.
