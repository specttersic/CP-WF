"""
消融研究损失函数

支持四种消融变体：
- full:      完整三重损失 (L_rec + L_per + L_con)  [基准]
- wo_per:    移除感知损失 (L_rec + L_con)
- wo_con:    移除对比损失 (L_rec + L_per)
- only_rec:  仅重建损失  (L_rec)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AblationLoss(nn.Module):
    """
    消融研究损失函数，通过 mode 参数控制使用哪些损失项。

    Args:
        mode: 消融模式
            'full'     - L_rec + L_per + L_con  (完整模型)
            'wo_per'   - L_rec + L_con           (移除感知损失)
            'wo_con'   - L_rec + L_per           (移除对比损失)
            'only_rec' - L_rec                   (仅重建损失)
        alpha:       L_rec 权重
        beta:        L_per 权重
        gamma:       L_con 权重
        temperature: SupCon 温度参数
    """

    MODES = ['full', 'wo_per', 'wo_con', 'only_rec']

    def __init__(self, mode='full', alpha=0.5, beta=0.25, gamma=0.25, temperature=0.5):
        super().__init__()
        assert mode in self.MODES, f"mode 必须是 {self.MODES} 之一，当前: {mode}"
        self.mode = mode
        self.temperature = temperature

        # 根据模式确定实际权重
        if mode == 'full':
            self.alpha, self.beta, self.gamma = alpha, beta, gamma
        elif mode == 'wo_per':
            # 去掉 L_per，将其权重分给 L_rec
            self.alpha = alpha + beta
            self.beta = 0.0
            self.gamma = gamma
        elif mode == 'wo_con':
            # 去掉 L_con，将其权重分给 L_rec
            self.alpha = alpha + gamma
            self.beta = beta
            self.gamma = 0.0
        elif mode == 'only_rec':
            self.alpha = 1.0
            self.beta = 0.0
            self.gamma = 0.0

        print(f"\nAblationLoss [{mode}] initialized:")
        print(f"  alpha (L_rec): {self.alpha}")
        print(f"  beta  (L_per): {self.beta}")
        print(f"  gamma (L_con): {self.gamma}")

    # ------------------------------------------------------------------
    def _perceptual_loss(self, feat_purified, feat_clean):
        """特征空间 L2 距离（感知损失）"""
        return F.mse_loss(feat_purified, feat_clean)

    def _supcon_loss(self, feat_purified, feat_clean, labels):
        """监督对比损失（SupCon）"""
        batch_size = feat_purified.shape[0]

        f1 = F.normalize(feat_purified, dim=1)
        f2 = F.normalize(feat_clean,    dim=1)

        features = torch.cat([f1, f2], dim=0)           # (2B, D)
        labels2  = torch.cat([labels, labels], dim=0)   # (2B,)

        sim = torch.matmul(features, features.T) / self.temperature  # (2B, 2B)

        labels2 = labels2.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels2, labels2.T).float()              # (2B, 2B)

        # 去除对角线自身
        eye_mask = 1 - torch.eye(2 * batch_size, device=sim.device)
        pos_mask = pos_mask * eye_mask

        exp_sim   = torch.exp(sim) * eye_mask
        log_prob  = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)
        mean_pos  = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + 1e-8)

        return -mean_pos.mean()

    # ------------------------------------------------------------------
    def forward(self, purified, clean, labels, encoder):
        """
        Args:
            purified: (B, 1, 1, 512) 净化流量
            clean:    (B, 1, 1, 512) 原始流量
            labels:   (B,)           类别标签
            encoder:  冻结的独立编码器

        Returns:
            total_loss: 标量
            loss_dict:  各项损失值字典
        """
        # 1. 重建损失（始终计算）
        loss_rec = F.mse_loss(purified, clean)

        loss_dict = {
            'mse': loss_rec.item(),
            'perceptual': 0.0,
            'supcon': 0.0,
        }

        total_loss = self.alpha * loss_rec

        # 2. 需要 encoder 特征的损失项
        if self.beta > 0 or self.gamma > 0:
            purified_seq = purified.squeeze()   # (B, 512)
            clean_seq    = clean.squeeze()

            feat_purified = encoder(purified_seq)
            with torch.no_grad():
                feat_clean = encoder(clean_seq)

            if self.beta > 0:
                loss_per = self._perceptual_loss(feat_purified, feat_clean)
                total_loss = total_loss + self.beta * loss_per
                loss_dict['perceptual'] = loss_per.item()

            if self.gamma > 0:
                loss_con = self._supcon_loss(feat_purified, feat_clean, labels)
                total_loss = total_loss + self.gamma * loss_con
                loss_dict['supcon'] = loss_con.item()

        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict
