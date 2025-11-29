"""
VAR蒸馏模块

实现基于知识蒸馏的VAR模型训练，支持：
1. 普通蒸馏 (Normal Distillation)
2. 尺度感知蒸馏 (Scale-Aware Distillation)

作者：Claude
日期：2024
"""

from .distillation_losses import DistillationLosses
from .distillation_trainer import DistillationVARTrainer
from .distill_utils import load_teacher_model, count_parameters, parse_scale_weights

__all__ = [
    'DistillationLosses',
    'DistillationVARTrainer',
    'load_teacher_model',
    'count_parameters',
    'parse_scale_weights'
]

__version__ = '1.0.0'