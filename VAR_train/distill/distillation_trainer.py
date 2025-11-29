"""
蒸馏训练器

继承VARTrainer，添加知识蒸馏功能。
保持原有的所有功能：渐进式训练、位置权重、DDP等。

支持的蒸馏方法：
1. 普通蒸馏 (Normal Distillation)
2. 尺度感知蒸馏 (Scale-Aware Distillation)
"""

import time
import torch
import torch.nn as nn
from typing import Tuple, Optional, Union

import dist
from trainer import VARTrainer
from distillation_losses import DistillationLosses
from distill_utils import count_parameters, validate_teacher_student_compatibility

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor


class DistillationVARTrainer(VARTrainer):
    """蒸馏训练器

    继承VARTrainer，添加知识蒸馏功能。
    保持原有的所有功能：渐进式训练、位置权重、DDP等。
    """

    def __init__(self, args, teacher_model=None, **kwargs):
        """初始化蒸馏训练器

        Args:
            args: 包含蒸馏配置的参数对象
            teacher_model: 预训练的教师模型
            **kwargs: 传递给父类VARTrainer的参数
        """
        super().__init__(**kwargs)

        self.args = args
        self.teacher_model = teacher_model
        self.distill_losses = DistillationLosses(args)

        # 蒸馏统计信息
        self.distill_stats = {
            'total_steps': 0,
            'avg_task_loss': 0.0,
            'avg_distill_loss': 0.0,
            'avg_logits_similarity': 0.0
        }

        # 冻结教师模型
        if self.teacher_model:
            self.teacher_model.eval()
            frozen_params = 0
            for param in self.teacher_model.parameters():
                param.requires_grad = False
                frozen_params += param.numel()

            teacher_param_count = count_parameters(self.teacher_model)
            print(f"[蒸馏训练器] 教师模型已冻结")
            print(f"[蒸馏训练器] 教师参数: {teacher_param_count:.2f}M ({frozen_params/1e6:.2f}M已冻结)")

            # 验证教师-学生兼容性
            student_model = kwargs.get('var_wo_ddp')
            if student_model:
                validate_teacher_student_compatibility(self.teacher_model, student_model, args)

        print(f"[蒸馏训练器] 初始化完成")
        print(f"[蒸馏训练器] 蒸馏类型: {args.distill_type}")
        print(f"[蒸馏训练器] 损失权重: 任务={args.distill_alpha:.2f}, 蒸馏={args.distill_beta:.2f}")

    def train_step(
        self, it: int, g_it: int, stepping: bool, metric_lg, tb_lg,
        inp_B3HW: FTen, label_B: Union[ITen, FTen], prog_si: int, prog_wp_it: float,
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        """重写训练步骤，添加蒸馏逻辑

        完全保持VARTrainer的原有逻辑，在适当位置添加蒸馏功能。
        """
        # ========== 1. 渐进式训练状态更新（保持原有逻辑） ==========
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = prog_si
        if self.last_prog_si != prog_si:
            if self.last_prog_si != -1:
                self.first_prog = False
            self.last_prog_si = prog_si
            self.prog_it = 0
        self.prog_it += 1
        prog_wp = max(min(self.prog_it / prog_wp_it, 1), 0.01)
        if self.first_prog:
            prog_wp = 1    # no prog warmup at first prog stage, as it's already solved in wp
        if prog_si == len(self.patch_nums) - 1:
            prog_si = -1    # max prog, as if no prog

        # ========== 2. 数据预处理（保持原有逻辑） ==========
        B, V = label_B.shape[0], self.vae_local.vocab_size
        self.var.require_backward_grad_sync = stepping

        gt_idx_Bl = self.vae_local.img_to_idxBl(inp_B3HW)
        gt_BL = torch.cat(gt_idx_Bl, dim=1)
        x_BLCv_wo_first_l = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)

        with self.var_opt.amp_ctx:
            # ========== 3. 学生模型前向（保持原有逻辑） ==========
            student_logits_BLV = self.var(label_B, x_BLCv_wo_first_l)

            # ========== 4. 教师模型前向（新增蒸馏逻辑） ==========
            teacher_logits_BLV = None
            if self.teacher_model is not None:
                with torch.no_grad():
                    # 渐进式训练时，需要截取教师模型的输入以匹配学生的token数量
                    # 学生模型：x_BLCv_wo_first_l shape = (B, student_L, Cvae)
                    # 需要确保教师模型处理相同数量的token
                    student_seq_len = x_BLCv_wo_first_l.shape[1]  # 学生的序列长度（不含first_l）

                    # 截取教师模型的输入token，使其与学生的序列长度一致
                    teacher_input = x_BLCv_wo_first_l[:, :student_seq_len, :]

                    # 临时设置教师模型的prog_si，确保其处理相同数量的token
                    original_teacher_prog_si = self.teacher_model.prog_si
                    if prog_si >= 0:
                        # 渐进式训练：教师也使用相同的prog_si
                        self.teacher_model.prog_si = prog_si

                    teacher_logits_BLV = self.teacher_model(label_B, teacher_input)

                    # 恢复教师模型的prog_si
                    self.teacher_model.prog_si = original_teacher_prog_si

                    # 截取教师logits以匹配学生logits的长度
                    # 学生logits: (B, student_ed, V), 教师logits可能更长
                    if teacher_logits_BLV.shape[1] > student_logits_BLV.shape[1]:
                        teacher_logits_BLV = teacher_logits_BLV[:, :student_logits_BLV.shape[1], :]

            # ========== 5. 计算任务损失（完全保持VAR原有逻辑） ==========
            task_loss = self.train_loss(student_logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)

            # 应用位置权重（完全保持VAR原有逻辑）
            if prog_si >= 0:    # 渐进式训练
                bg, ed = self.begin_ends[prog_si]
                assert student_logits_BLV.shape[1] == gt_BL.shape[1] == ed
                lw = self.loss_weight[:, :ed].clone()
                lw[:, bg:ed] *= min(max(prog_wp, 0), 1)
            else:               # 普通训练
                lw = self.loss_weight

            task_loss = task_loss.mul(lw).sum(dim=-1).mean()

            # ========== 6. 计算蒸馏损失（新增蒸馏逻辑） ==========
            if teacher_logits_BLV is not None:
                distill_loss = self.distill_losses.compute_loss(
                    student_logits_BLV, teacher_logits_BLV, prog_si
                )

                # 组合总损失
                total_loss = (self.args.distill_alpha * task_loss +
                             self.args.distill_beta * distill_loss)

                # 更新蒸馏统计信息
                self._update_distill_stats(task_loss, distill_loss, student_logits_BLV, teacher_logits_BLV)
            else:
                total_loss = task_loss
                distill_loss = torch.tensor(0.0, device=task_loss.device)

        # ========== 7. 反向传播（保持原有逻辑） ==========
        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=total_loss, stepping=stepping)

        # ========== 8. 日志记录（扩展原有逻辑） ==========
        self._log_distillation_metrics(
            it, g_it, metric_lg, tb_lg, student_logits_BLV, teacher_logits_BLV,
            gt_BL, V, task_loss, distill_loss, prog_si, prog_wp, prog_wp_it
        )

        # ========== 9. 重置渐进式训练状态（保持原有逻辑） ==========
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = -1

        return grad_norm, scale_log2

    def _update_distill_stats(self, task_loss: torch.Tensor, distill_loss: torch.Tensor,
                             student_logits: torch.Tensor, teacher_logits: torch.Tensor):
        """更新蒸馏统计信息"""
        self.distill_stats['total_steps'] += 1
        step = self.distill_stats['total_steps']

        # 计算移动平均
        alpha = 0.99  # 移动平均系数
        self.distill_stats['avg_task_loss'] = (
            alpha * self.distill_stats['avg_task_loss'] +
            (1 - alpha) * task_loss.item()
        )
        self.distill_stats['avg_distill_loss'] = (
            alpha * self.distill_stats['avg_distill_loss'] +
            (1 - alpha) * distill_loss.item()
        )

        # 计算logits相似度（每100步一次，避免过多计算）
        if step % 100 == 0:
            similarity = self.distill_losses.compute_logits_similarity(student_logits, teacher_logits)
            self.distill_stats['avg_logits_similarity'] = (
                alpha * self.distill_stats['avg_logits_similarity'] +
                (1 - alpha) * similarity.item()
            )

    def _log_distillation_metrics(self, it: int, g_it: int, metric_lg, tb_lg,
                                 student_logits: torch.Tensor, teacher_logits: Optional[torch.Tensor],
                                 gt_BL: torch.Tensor, V: int, task_loss: torch.Tensor,
                                 distill_loss: torch.Tensor, prog_si: int, prog_wp: float, prog_wp_it: float):
        """扩展的日志记录，添加蒸馏指标

        保持VARTrainer原有的所有日志记录逻辑，添加蒸馏相关指标。
        """
        pred_BL = student_logits.data.argmax(dim=-1)

        # ========== 基础指标记录（保持原有逻辑） ==========
        if it == 0 or it in metric_lg.log_iters:
            Lmean = self.val_loss(student_logits.data.view(-1, V), gt_BL.view(-1)).item()
            acc_mean = (pred_BL == gt_BL).float().mean().item() * 100

            if prog_si >= 0:    # 渐进式训练
                Ltail = acc_tail = -1
            else:               # 普通训练
                Ltail = self.val_loss(student_logits.data[:, -self.last_l:].reshape(-1, V),
                                    gt_BL[:, -self.last_l:].reshape(-1)).item()
                acc_tail = (pred_BL[:, -self.last_l:] == gt_BL[:, -self.last_l:]).float().mean().item() * 100

            grad_norm_val = 0.0  # 这里会在调用方设置

            # 蒸馏指标
            task_loss_val = task_loss.item()
            distill_loss_val = distill_loss.item() if isinstance(distill_loss, torch.Tensor) else 0.0
            total_loss_val = task_loss_val + distill_loss_val

            # 更新指标记录器（添加蒸馏指标）
            metric_lg.update(Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail,
                           tnm=grad_norm_val,  # 这个会在返回后被正确设置
                           TaskL=task_loss_val, DistillL=distill_loss_val, TotalL=total_loss_val)

        # ========== Tensorboard记录（扩展原有逻辑） ==========
        if g_it == 0 or (g_it + 1) % 500 == 0:
            # 原有指标记录（保持不变）
            prob_per_class_is_chosen = pred_BL.view(-1).bincount(minlength=V).float()
            dist.allreduce(prob_per_class_is_chosen)
            prob_per_class_is_chosen /= prob_per_class_is_chosen.sum()
            cluster_usage = (prob_per_class_is_chosen > 0.001 / V).float().mean().item() * 100

            if dist.is_master():
                if g_it == 0:
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-10000)
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-1000)

                # 原有的分尺度精度记录
                kw = dict(z_voc_usage=cluster_usage)
                for si, (bg, ed) in enumerate(self.begin_ends):
                    if 0 <= prog_si < si:
                        break
                    pred, tar = student_logits.data[:, bg:ed].reshape(-1, V), gt_BL[:, bg:ed].reshape(-1)
                    acc = (pred.argmax(dim=-1) == tar).float().mean().item() * 100
                    ce = self.val_loss(pred, tar).item()
                    kw[f'acc_{self.resos[si]}'] = acc
                    kw[f'L_{self.resos[si]}'] = ce

                tb_lg.update(head='AR_iter_loss', **kw, step=g_it)
                tb_lg.update(head='AR_iter_schedule',
                           prog_a_reso=self.resos[prog_si] if prog_si >= 0 else self.resos[-1],
                           prog_si=prog_si, prog_wp=prog_wp, step=g_it)

                # 新增蒸馏损失记录
                distill_kw = {
                    'task_loss': task_loss.item(),
                    'distill_loss': distill_loss.item() if isinstance(distill_loss, torch.Tensor) else 0.0,
                    'total_loss': (task_loss + distill_loss).item() if isinstance(distill_loss, torch.Tensor) else task_loss.item(),
                }

                # 添加logits相似度（如果可用）
                if teacher_logits is not None and (g_it + 1) % 1000 == 0:  # 每1000步计算一次相似度
                    similarity = self.distill_losses.compute_logits_similarity(student_logits, teacher_logits)
                    distill_kw['logits_similarity'] = similarity.item()

                tb_lg.update(head='Distill_loss', **distill_kw, step=g_it)

                # 记录尺度信息（如果是尺度感知蒸馏）
                if self.args.distill_type in ['scale_aware', 'both'] and (g_it + 1) % 2000 == 0:
                    scale_info = self.distill_losses.get_scale_info(prog_si)
                    tb_lg.update(head='Distill_scale_info',
                               active_scales=scale_info['active_scales'],
                               total_tokens=scale_info['total_tokens'],
                               step=g_it)

    @torch.no_grad()
    def eval_ep(self, ld_val):
        """重写验证方法，添加蒸馏评估

        保持原有验证逻辑，添加蒸馏相关指标的记录。
        """
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        distill_loss_total = 0
        logits_similarity_total = 0

        stt = time.time()
        training = self.var_wo_ddp.training
        self.var_wo_ddp.eval()

        for inp_B3HW, label_B in ld_val:
            B, V = label_B.shape[0], self.vae_local.vocab_size
            inp_B3HW = inp_B3HW.to(dist.get_device(), non_blocking=True)
            label_B = label_B.to(dist.get_device(), non_blocking=True)

            gt_idx_Bl = self.vae_local.img_to_idxBl(inp_B3HW)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            x_BLCv_wo_first_l = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)

            # 学生模型前向
            student_logits_BLV = self.var_wo_ddp(label_B, x_BLCv_wo_first_l)

            # 原有指标计算
            L_mean += self.val_loss(student_logits_BLV.data.view(-1, V), gt_BL.view(-1)) * B
            L_tail += self.val_loss(student_logits_BLV.data[:, -self.last_l:].reshape(-1, V),
                                  gt_BL[:, -self.last_l:].reshape(-1)) * B
            acc_mean += (student_logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (100/gt_BL.shape[1])
            acc_tail += (student_logits_BLV.data[:, -self.last_l:].argmax(dim=-1) == gt_BL[:, -self.last_l:]).sum() * (100 / self.last_l)

            # 蒸馏相关指标
            if self.teacher_model is not None:
                with torch.no_grad():
                    teacher_logits_BLV = self.teacher_model(label_B, x_BLCv_wo_first_l)
                    distill_loss = self.distill_losses.compute_loss(student_logits_BLV, teacher_logits_BLV, -1)
                    distill_loss_total += distill_loss.item() * B

                    # 每10个batch计算一次相似度（避免过多计算）
                    if tot % 10 == 0:
                        similarity = self.distill_losses.compute_logits_similarity(student_logits_BLV, teacher_logits_BLV)
                        logits_similarity_total += similarity.item() * B

            tot += B

        self.var_wo_ddp.train(training)

        # 聚合所有GPU的统计信息
        stats = L_mean.new_tensor([L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(),
                                  distill_loss_total, logits_similarity_total, tot])
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        L_mean, L_tail, acc_mean, acc_tail, avg_distill_loss, avg_similarity, _ = stats.tolist()

        eval_time = time.time() - stt

        # 打印蒸馏相关信息
        if self.teacher_model is not None and dist.is_master():
            print(f"[验证蒸馏] 蒸馏损失: {avg_distill_loss:.4f}, Logits相似度: {avg_similarity:.4f}")

        return L_mean, L_tail, acc_mean, acc_tail, tot, eval_time

    def get_distill_stats(self):
        """获取蒸馏统计信息"""
        return self.distill_stats.copy()

    def get_config(self):
        """扩展配置信息，添加蒸馏参数"""
        config = super().get_config()
        if hasattr(self, 'args'):
            config.update({
                'distill_type': self.args.distill_type,
                'distill_alpha': self.args.distill_alpha,
                'distill_beta': self.args.distill_beta,
                'distill_temperature': self.args.distill_temperature,
                'scale_weights': self.args.scale_weights,
                'teacher_depth': self.args.teacher_depth,
            })
        return config