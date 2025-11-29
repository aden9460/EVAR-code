"""
蒸馏损失函数

实现两种核心蒸馏方法：
1. 普通蒸馏 (Normal Distillation): 统一KL散度损失
2. 尺度感知蒸馏 (Scale-Aware Distillation): 分尺度加权KL散度损失

支持VAR的10个尺度结构和渐进式训练。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class DistillationLosses:
    """蒸馏损失计算类

    支持两种蒸馏方法：
    1. 普通蒸馏：对所有token位置统一应用KL散度
    2. 尺度感知蒸馏：按VAR的10个尺度分别计算KL散度，应用不同权重
    """

    def __init__(self, args):
        """初始化蒸馏损失计算器

        Args:
            args: 包含蒸馏配置的参数对象
        """
        self.temperature = args.distill_temperature
        self.distill_type = args.distill_type
        self.scale_weights = args.scale_weights

        # VAR的10个尺度边界 (累积token数量)
        # 尺度0: 1×1=1, 尺度1: 2×2=4, 尺度2: 3×3=9, ..., 尺度9: 16×16=256
        # 累积: [0, 1, 5, 14, 30, 55, 91, 155, 255, 424, 680]
        self.scale_boundaries = [0, 1, 5, 14, 30, 55, 91, 155, 255, 424, 680]

        # 验证配置
        assert len(self.scale_weights) == 10, f"尺度权重必须有10个值，当前有{len(self.scale_weights)}个"
        assert self.temperature > 0, "蒸馏温度必须大于0"
        assert self.distill_type in ['normal', 'scale_aware', 'both'], f"不支持的蒸馏类型: {self.distill_type}"

        print(f"[蒸馏损失] 类型={self.distill_type}, 温度={self.temperature}")
        print(f"[尺度权重] {self.scale_weights}")

    def compute_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, prog_si: int = -1) -> torch.Tensor:
        """计算蒸馏损失

        Args:
            student_logits: [B, L, V] 学生模型输出logits
            teacher_logits: [B, L, V] 教师模型输出logits
            prog_si: 渐进式训练当前尺度索引 (-1表示全尺度训练)

        Returns:
            蒸馏损失值

        Raises:
            ValueError: 不支持的蒸馏类型
        """
        # 验证输入形状
        assert student_logits.shape == teacher_logits.shape, \
            f"学生和教师logits形状不匹配: {student_logits.shape} vs {teacher_logits.shape}"

        B, L, V = student_logits.shape

        # 验证序列长度（支持渐进式训练）
        if prog_si >= 0:
            # 渐进式训练：验证长度匹配当前阶段
            expected_L = self.scale_boundaries[prog_si + 1]
            if L != expected_L:
                print(f"[警告] 渐进式训练阶段{prog_si}的序列长度不匹配: 期望{expected_L}，实际{L}")
        else:
            # 全尺度训练：期望完整长度680
            if L != 680:
                print(f"[警告] 全尺度训练的序列长度不是680: 实际{L}")

        if self.distill_type == 'normal':
            return self._normal_distill_loss(student_logits, teacher_logits, prog_si)
        elif self.distill_type == 'scale_aware':
            return self._scale_aware_distill_loss(student_logits, teacher_logits, prog_si)
        elif self.distill_type == 'both':
            normal_loss = self._normal_distill_loss(student_logits, teacher_logits, prog_si)
            scale_loss = self._scale_aware_distill_loss(student_logits, teacher_logits, prog_si)
            return 0.5 * normal_loss + 0.5 * scale_loss
        else:
            raise ValueError(f"不支持的蒸馏类型: {self.distill_type}")

    def _normal_distill_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, prog_si: int) -> torch.Tensor:
        """普通蒸馏损失：统一KL散度

        Args:
            student_logits: [B, L, V] 学生模型logits
            teacher_logits: [B, L, V] 教师模型logits
            prog_si: 渐进式训练当前尺度（用于确定有效长度）

        Returns:
            KL散度损失
        """
        # 如果是渐进式训练，只计算到当前尺度
        if prog_si >= 0:
            effective_length = self.scale_boundaries[prog_si + 1]
            student_logits = student_logits[:, :effective_length, :]
            teacher_logits = teacher_logits[:, :effective_length, :]

        # 计算软化的概率分布
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_soft = F.log_softmax(student_logits / self.temperature, dim=-1)

        # 计算KL散度
        kl_loss = F.kl_div(student_log_soft, teacher_soft, reduction='batchmean')

        # 温度平方补偿（标准做法）
        return kl_loss * (self.temperature ** 2)

    def _scale_aware_distill_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, prog_si: int) -> torch.Tensor:
        """尺度感知蒸馏损失：分尺度加权KL散度

        Args:
            student_logits: [B, L, V] 学生模型logits
            teacher_logits: [B, L, V] 教师模型logits
            prog_si: 渐进式训练当前尺度索引

        Returns:
            加权KL散度损失
        """
        total_loss = 0.0
        num_scales = 0

        # 确定训练的尺度范围
        if prog_si >= 0:
            # 渐进式训练：只训练到当前尺度
            scale_range = range(prog_si + 1)
        else:
            # 全尺度训练
            scale_range = range(10)

        for i in scale_range:
            start_pos = self.scale_boundaries[i]
            end_pos = self.scale_boundaries[i + 1]

            # 提取该尺度的logits
            teacher_scale = teacher_logits[:, start_pos:end_pos, :]  # [B, scale_tokens, V]
            student_scale = student_logits[:, start_pos:end_pos, :]

            # 跳过空尺度（理论上不应该发生）
            if start_pos >= end_pos:
                continue

            # 计算该尺度的KL散度
            teacher_soft = F.softmax(teacher_scale / self.temperature, dim=-1)
            student_log_soft = F.log_softmax(student_scale / self.temperature, dim=-1)
            scale_kl = F.kl_div(student_log_soft, teacher_soft, reduction='batchmean')

            # 应用尺度权重
            weight = self.scale_weights[i] if i < len(self.scale_weights) else 1.0
            total_loss += weight * scale_kl
            num_scales += 1

        # 避免除零
        if num_scales == 0:
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True)

        # 温度平方补偿
        return total_loss * (self.temperature ** 2)

    def compute_logits_similarity(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """计算学生和教师logits的余弦相似度

        Args:
            student_logits: [B, L, V] 学生模型logits
            teacher_logits: [B, L, V] 教师模型logits

        Returns:
            平均余弦相似度
        """
        # 展平为 [B*L, V]
        student_flat = student_logits.view(-1, student_logits.size(-1))
        teacher_flat = teacher_logits.view(-1, teacher_logits.size(-1))

        # 计算余弦相似度
        cosine_sim = F.cosine_similarity(student_flat, teacher_flat, dim=-1)
        return cosine_sim.mean()

    def get_scale_info(self, prog_si: int = -1) -> dict:
        """获取当前训练的尺度信息

        Args:
            prog_si: 渐进式训练当前尺度索引

        Returns:
            尺度信息字典
        """
        if prog_si >= 0:
            active_scales = prog_si + 1
            total_tokens = self.scale_boundaries[prog_si + 1]
        else:
            active_scales = 10
            total_tokens = 680

        scale_info = {
            'active_scales': active_scales,
            'total_scales': 10,
            'total_tokens': total_tokens,
            'scale_boundaries': self.scale_boundaries[:active_scales + 1],
            'active_weights': self.scale_weights[:active_scales]
        }

        return scale_info

    def __repr__(self):
        return (f"DistillationLosses(type={self.distill_type}, "
                f"temperature={self.temperature}, "
                f"scale_weights={self.scale_weights})")


class FeatureDistillationLoss(nn.Module):
    """特征蒸馏损失（未来扩展）

    在中间层进行特征匹配，传递更丰富的语义信息。
    """

    def __init__(self, feature_layers: List[int] = [4, 8, 12, 15]):
        """初始化特征蒸馏损失

        Args:
            feature_layers: 需要进行特征蒸馏的层索引
        """
        super().__init__()
        self.feature_layers = feature_layers
        self.projections = nn.ModuleDict()  # 维度对齐层

        print(f"[特征蒸馏] 目标层: {feature_layers}")

    def forward(self, student_features: List[torch.Tensor], teacher_features: List[torch.Tensor]) -> torch.Tensor:
        """计算特征蒸馏损失

        Args:
            student_features: 学生模型中间特征列表
            teacher_features: 教师模型中间特征列表

        Returns:
            特征匹配损失
        """
        feature_loss = 0.0
        num_layers = 0

        for layer_idx in self.feature_layers:
            if layer_idx >= len(student_features) or layer_idx >= len(teacher_features):
                continue

            student_feat = student_features[layer_idx]  # [B, L, C_student]
            teacher_feat = teacher_features[layer_idx]  # [B, L, C_teacher]

            # 维度对齐（如果教师和学生维度不同）
            if student_feat.shape[-1] != teacher_feat.shape[-1]:
                proj_name = f"proj_{layer_idx}"
                if proj_name not in self.projections:
                    self.projections[proj_name] = nn.Linear(
                        student_feat.shape[-1], teacher_feat.shape[-1]
                    ).to(student_feat.device)
                student_feat = self.projections[proj_name](student_feat)

            # 计算特征匹配损失
            feat_loss = F.mse_loss(student_feat, teacher_feat)
            feature_loss += feat_loss
            num_layers += 1

        return feature_loss / max(num_layers, 1)


class AttentionDistillationLoss(nn.Module):
    """注意力蒸馏损失（未来扩展）

    匹配教师和学生模型的注意力图，传递关注机制。
    """

    def __init__(self, attention_layers: List[int] = [8, 12, 15]):
        """初始化注意力蒸馏损失

        Args:
            attention_layers: 需要进行注意力蒸馏的层索引
        """
        super().__init__()
        self.attention_layers = attention_layers

        print(f"[注意力蒸馏] 目标层: {attention_layers}")

    def forward(self, student_attentions: List[torch.Tensor], teacher_attentions: List[torch.Tensor]) -> torch.Tensor:
        """计算注意力蒸馏损失

        Args:
            student_attentions: 学生模型注意力图列表, 每个形状 [B, num_heads, L, L]
            teacher_attentions: 教师模型注意力图列表, 每个形状 [B, num_heads, L, L]

        Returns:
            注意力匹配损失
        """
        attention_loss = 0.0
        num_layers = 0

        for layer_idx in self.attention_layers:
            if layer_idx >= len(student_attentions) or layer_idx >= len(teacher_attentions):
                continue

            student_attn = student_attentions[layer_idx]  # [B, H_s, L, L]
            teacher_attn = teacher_attentions[layer_idx]  # [B, H_t, L, L]

            # 如果注意力头数不同，需要处理
            if student_attn.shape[1] != teacher_attn.shape[1]:
                # 方法1: 平均池化教师注意力
                teacher_attn = teacher_attn.mean(dim=1, keepdim=True)  # [B, 1, L, L]
                teacher_attn = teacher_attn.expand(-1, student_attn.shape[1], -1, -1)

            # 计算注意力图的MSE损失
            attn_loss = F.mse_loss(student_attn, teacher_attn)
            attention_loss += attn_loss
            num_layers += 1

        return attention_loss / max(num_layers, 1)