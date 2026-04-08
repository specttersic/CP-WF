"""
改进的损失函数 - 局部-全局对比学习

核心改进：
1. 用局部-全局对比学习替换MSE损失
2. 更细粒度地确保净化后的流量与原始流量接近
3. 同时在信号级和特征级进行约束

损失函数组成：
- L_global: 全局对比学习（整个序列）
- L_local: 局部对比学习（4个段）
- L_cons: 一致性约束（局部聚合 vs 全局）
- L_feature: 特征级对比学习（使用encoder）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalGlobalContrastiveLoss(nn.Module):
    """
    局部-全局对比学习损失
    
    核心思想：
    1. 全局对比：确保整个序列在特征空间中接近
    2. 局部对比：确保每个段在统计空间中接近
    3. 一致性约束：确保局部和全局协调
    
    Args:
        w_global: 全局对比学习权重（默认0.5）
        w_local: 局部对比学习权重（默认0.3）
        w_cons: 一致性约束权重（默认0.2）
        num_segments: 分段数量（默认4）
        temperature: 对比学习温度（默认0.1）
    """
    
    def __init__(self, w_global=0.5, w_local=0.3, w_cons=0.2, 
                 num_segments=4, temperature=0.5):
        super().__init__()
        self.w_global = w_global
        self.w_local = w_local
        self.w_cons = w_cons
        self.num_segments = num_segments
        self.temperature = temperature
        
        # 确保权重和为1
        total = w_global + w_local + w_cons
        assert abs(total - 1.0) < 1e-6, f"权重和必须为1，当前为{total}"
        
        print(f"\nLocalGlobalContrastiveLoss initialized:")
        print(f"  - w_global: {w_global}")
        print(f"  - w_local: {w_local}")
        print(f"  - w_cons: {w_cons}")
        print(f"  - num_segments: {num_segments}")
        print(f"  - temperature: {temperature} (更大的温度以稳定训练)")
    
    def extract_local_features(self, x):
        """
        提取局部特征（分段统计特征）
        
        Args:
            x: (batch, 512) 流量序列
            
        Returns:
            local_features: (batch, num_segments, 2) 每段的[均值, 标准差]
        """
        batch_size = x.size(0)
        segment_size = 512 // self.num_segments  # 128
        
        local_features = []
        for i in range(self.num_segments):
            start = i * segment_size
            end = (i + 1) * segment_size
            segment = x[:, start:end]  # (batch, 128)
            
            # 计算统计特征
            mean = segment.mean(dim=1, keepdim=True)  # (batch, 1)
            std = segment.std(dim=1, keepdim=True)    # (batch, 1)
            
            seg_feat = torch.cat([mean, std], dim=1)  # (batch, 2)
            local_features.append(seg_feat)
        
        local_features = torch.stack(local_features, dim=1)  # (batch, 4, 2)
        return local_features
    
    def contrastive_loss(self, feat1, feat2, labels):
        """
        计算对比学习损失（配对样本）
        
        Args:
            feat1: (batch, dim) 净化样本特征
            feat2: (batch, dim) 原始样本特征
            labels: (batch,) 类别标签
            
        Returns:
            loss: 对比学习损失
        """
        batch_size = feat1.shape[0]
        
        # 归一化
        feat1 = F.normalize(feat1, dim=1)
        feat2 = F.normalize(feat2, dim=1)
        
        # 拼接特征
        features = torch.cat([feat1, feat2], dim=0)  # (2*batch, dim)
        labels = torch.cat([labels, labels], dim=0)  # (2*batch,)
        
        # 计算相似度矩阵
        similarity = torch.matmul(features, features.T) / self.temperature
        
        # 构建正样本mask（同类样本）
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # 去除对角线
        logits_mask = 1 - torch.eye(2*batch_size, device=mask.device)
        mask = mask * logits_mask
        
        # 计算损失
        exp_logits = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # 只对正样本计算平均
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = -mean_log_prob_pos.mean()
        
        return loss
    
    def forward(self, denoised, original, labels, encoder):
        """
        计算局部-全局对比学习损失
        
        Args:
            denoised: (batch, 1, 1, 512) 净化流量
            original: (batch, 1, 1, 512) 原始流量
            labels: (batch,) 类别标签
            encoder: 预训练的encoder
            
        Returns:
            total_loss: 总损失
            loss_dict: 各部分损失的字典
        """
        # 提取序列
        denoised_seq = denoised.squeeze()  # (batch, 512)
        original_seq = original.squeeze()  # (batch, 512)
        
        # 1. 全局对比学习（使用encoder提取特征）
        # 注意：不使用no_grad，让梯度可以回传到denoiser
        denoised_global = encoder(denoised_seq)  # (batch, 512)
        with torch.no_grad():
            original_global = encoder(original_seq)  # (batch, 512)
        
        loss_global = self.contrastive_loss(denoised_global, original_global, labels)
        
        # 2. 局部对比学习（分段统计特征）
        denoised_local = self.extract_local_features(denoised_seq)  # (batch, 4, 2)
        original_local = self.extract_local_features(original_seq)  # (batch, 4, 2)
        
        loss_local = 0
        for i in range(self.num_segments):
            seg_denoised = denoised_local[:, i, :]  # (batch, 2)
            seg_original = original_local[:, i, :]  # (batch, 2)
            loss_local += self.contrastive_loss(seg_denoised, seg_original, labels)
        loss_local = loss_local / self.num_segments
        
        # 3. 一致性约束（局部聚合 vs 全局统计）
        # 局部聚合：所有段的均值和标准差的平均
        denoised_local_agg = denoised_local.mean(dim=1)  # (batch, 2)
        original_local_agg = original_local.mean(dim=1)  # (batch, 2)
        
        # 全局统计
        denoised_global_stat = torch.stack([
            denoised_seq.mean(dim=1),
            denoised_seq.std(dim=1)
        ], dim=1)  # (batch, 2)
        
        original_global_stat = torch.stack([
            original_seq.mean(dim=1),
            original_seq.std(dim=1)
        ], dim=1)  # (batch, 2)
        
        loss_cons = F.mse_loss(denoised_local_agg, denoised_global_stat) + \
                    F.mse_loss(original_local_agg, original_global_stat)
        
        # 4. 总损失
        total_loss = (self.w_global * loss_global + 
                     self.w_local * loss_local + 
                     self.w_cons * loss_cons)
        
        loss_dict = {
            'total': total_loss.item(),
            'global': loss_global.item(),
            'local': loss_local.item(),
            'consistency': loss_cons.item()
        }
        
        return total_loss, loss_dict


class ImprovedTripleLoss(nn.Module):
    """
    改进的三重损失函数
    
    损失组成：
    - L_mse: 整体MSE损失（像素级重建）
    - L_perceptual: Perceptual Loss（特征级距离）
    - L_supcon: 监督对比学习损失（类别约束）
    
    Args:
        alpha: MSE权重（默认0.5）
        beta: Perceptual Loss权重（默认0.25）
        gamma: SupCon权重（默认0.25）
        temperature: 对比学习温度（默认0.5）
    """
    
    def __init__(self, alpha=0.5, beta=0.25, gamma=0.25, temperature=0.5):
        super().__init__()
        self.alpha = alpha          # MSE权重
        self.beta = beta            # Perceptual权重
        self.gamma = gamma          # SupCon权重
        self.temperature = temperature
        
        # 确保权重和为1
        total = alpha + beta + gamma
        assert abs(total - 1.0) < 1e-6, f"权重和必须为1，当前为{total}"
        
        print(f"\nImprovedTripleLoss initialized:")
        print(f"  - alpha (MSE): {alpha}")
        print(f"  - beta (Perceptual): {beta}")
        print(f"  - gamma (SupCon): {gamma}")
        print(f"  - temperature: {temperature}")
        print(f"\n损失设计：")
        print(f"  1. MSE（像素级重建约束）")
        print(f"  2. Perceptual Loss（特征空间约束）")
        print(f"  3. SupCon（类别约束）")
    
    def perceptual_loss(self, feat1, feat2):
        """
        计算Perceptual Loss（特征空间L2距离）
        
        理论依据：
        - 广泛用于图像重建任务（SRGAN, Pix2Pix等）
        - 在特征空间中度量距离，比像素空间更符合感知
        - 与SupCon互补：Perceptual是距离约束（配对），SupCon是对比约束（类别）
        
        Args:
            feat1: (batch, dim) 净化样本特征
            feat2: (batch, dim) 原始样本特征
            
        Returns:
            loss: Perceptual Loss（L2距离）
        """
        # 计算L2距离
        loss = F.mse_loss(feat1, feat2)
        return loss
    
    def cosine_loss(self, feat1, feat2):
        """
        计算余弦相似度损失（方案三：保留Cosine）
        
        Args:
            feat1: (batch, dim) 净化样本特征
            feat2: (batch, dim) 原始样本特征
            
        Returns:
            loss: 余弦相似度损失
        """
        # 归一化
        feat1 = F.normalize(feat1, dim=1)
        feat2 = F.normalize(feat2, dim=1)
        
        # 计算余弦相似度
        cosine_sim = (feat1 * feat2).sum(dim=1)
        
        # 损失：1 - 余弦相似度
        loss = (1 - cosine_sim).mean()
        
        return loss
    
    def supcon_loss(self, feat1, feat2, labels):
        """
        计算监督对比学习损失
        
        Args:
            feat1: (batch, dim) 净化样本特征
            feat2: (batch, dim) 原始样本特征
            labels: (batch,) 类别标签
            
        Returns:
            loss: 监督对比学习损失
        """
        batch_size = feat1.shape[0]
        
        # 归一化
        feat1 = F.normalize(feat1, dim=1)
        feat2 = F.normalize(feat2, dim=1)
        
        # 拼接特征
        features = torch.cat([feat1, feat2], dim=0)  # (2*batch, dim)
        labels = torch.cat([labels, labels], dim=0)  # (2*batch,)
        
        # 计算相似度矩阵
        similarity = torch.matmul(features, features.T) / self.temperature
        
        # 构建正样本mask
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # 去除对角线
        logits_mask = 1 - torch.eye(2*batch_size, device=mask.device)
        mask = mask * logits_mask
        
        # 计算损失
        exp_logits = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # 只对正样本计算平均
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = -mean_log_prob_pos.mean()
        
        return loss
    
    def forward(self, denoised, original, labels, encoder):
        """
        计算改进的三重损失
        
        Args:
            denoised: (batch, 1, 1, 512) 净化流量
            original: (batch, 1, 1, 512) 原始流量
            labels: (batch,) 类别标签
            encoder: 预训练的encoder
            
        Returns:
            total_loss: 总损失
            loss_dict: 各部分损失的字典
        """
        # 提取序列
        denoised_seq = denoised.squeeze()  # (batch, 512)
        original_seq = original.squeeze()  # (batch, 512)
        
        # 1. MSE损失（像素级重建）
        loss_mse = F.mse_loss(denoised, original)
        
        # 2. 提取encoder特征
        denoised_feat = encoder(denoised_seq)
        with torch.no_grad():
            original_feat = encoder(original_seq)
        
        # 3. Perceptual Loss（特征空间L2距离）
        loss_perceptual = self.perceptual_loss(denoised_feat, original_feat)
        
        # 4. SupCon损失（类别约束）
        loss_supcon = self.supcon_loss(denoised_feat, original_feat, labels)
        
        # 5. 总损失
        total_loss = (self.alpha * loss_mse + 
                     self.beta * loss_perceptual + 
                     self.gamma * loss_supcon)
        
        loss_dict = {
            'total': total_loss.item(),
            'mse': loss_mse.item(),
            'perceptual': loss_perceptual.item(),
            'supcon': loss_supcon.item()
        }
        
        return total_loss, loss_dict


# ========== 测试代码 ==========
if __name__ == '__main__':
    """测试改进的损失函数"""
    
    print("=" * 60)
    print("测试 LocalGlobalContrastiveLoss")
    print("=" * 60)
    
    # 创建模拟encoder
    class MockEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(512, 512)
        
        def forward(self, x):
            return self.fc(x)
    
    encoder = MockEncoder()
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    
    # 创建损失函数
    criterion = LocalGlobalContrastiveLoss()
    
    # 测试数据
    batch_size = 32
    num_classes = 100
    denoised = torch.randn(batch_size, 1, 1, 512, requires_grad=True)
    original = torch.randn(batch_size, 1, 1, 512)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # 计算损失
    total_loss, loss_dict = criterion(denoised, original, labels, encoder)
    
    print(f"\n✓ LocalGlobalContrastiveLoss test passed!")
    print(f"  Batch size: {batch_size}")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Global loss: {loss_dict['global']:.4f}")
    print(f"  Local loss: {loss_dict['local']:.4f}")
    print(f"  Consistency loss: {loss_dict['consistency']:.4f}")
    
    # 测试梯度
    total_loss.backward()
    print(f"  Gradient computed: ✓")
    print(f"  Denoised has gradient: {denoised.grad is not None}")
    
    print("\n" + "=" * 60)
    print("测试 ImprovedTripleLoss")
    print("=" * 60)
    
    # 创建损失函数（不需要classifier）
    criterion2 = ImprovedTripleLoss(
        alpha=0.3,
        beta=0.2,
        gamma=0.25,
        delta=0.25
    )
    
    # 测试数据
    denoised = torch.randn(batch_size, 1, 1, 512, requires_grad=True)
    original = torch.randn(batch_size, 1, 1, 512)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # 计算损失（不需要classifier参数）
    total_loss, loss_dict = criterion2(denoised, original, labels, encoder)
    
    print(f"\n✓ ImprovedTripleLoss test passed!")
    print(f"  Batch size: {batch_size}")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  MSE整体: {loss_dict['mse_global']:.4f}")
    print(f"  MSE分段: {loss_dict['mse_local']:.4f}")
    print(f"  Perceptual: {loss_dict['perceptual']:.4f}")
    print(f"  SupCon: {loss_dict['supcon']:.4f}")
    
    # 测试梯度
    total_loss.backward()
    print(f"  Gradient computed: ✓")
    print(f"  Denoised has gradient: {denoised.grad is not None}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)

