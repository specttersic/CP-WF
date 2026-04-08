"""
# 混合编码器：分类器特征（30%）+ 独立编码器（70%）

改进版本：
- 使用1D卷积提取时序特征
- Xavier初始化
- 渐进式通道扩张（32→64→128→256）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    """1D双卷积块（借鉴NetCLR）"""
    
    def __init__(self, in_channels, out_channels, kernel_size=8, 
                 pool_size=4, pool_stride=2, dropout=0.1):
        super().__init__()
        
        # 第一个卷积
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               stride=1, padding='same')
        
        # 第二个卷积
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                               stride=1, padding='same')
        
        # BatchNorm（放在第二个卷积后）
        self.bn = nn.BatchNorm1d(out_channels)
        
        # MaxPool
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Xavier初始化
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初始化"""
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
    
    def forward(self, x):
        # 第一个卷积 + ELU
        x = F.elu(self.conv1(x))
        
        # 第二个卷积 + BatchNorm + ELU
        x = F.elu(self.bn(self.conv2(x)))
        
        # MaxPool + Dropout
        x = self.pool(x)
        x = self.dropout(x)
        
        return x


class IndependentEncoder(nn.Module):
    """
    独立编码器（使用1D卷积）
    
    输入：(batch, 512) burst序列
    输出：(batch, 512) 嵌入特征
    
    架构：
    - 4个双卷积块（32 → 64 → 128 → 256）
    - 全局平均池化
    - 全连接层投影到512维
    """
    
    def __init__(self, input_dim=512, output_dim=512):
        super().__init__()
        
        # Block 1: 1 → 32
        self.block1 = ConvBlock1D(
            in_channels=1, 
            out_channels=32,
            kernel_size=8,
            pool_size=4,
            pool_stride=2,
            dropout=0.1
        )
        
        # Block 2: 32 → 64
        self.block2 = ConvBlock1D(
            in_channels=32,
            out_channels=64,
            kernel_size=8,
            pool_size=4,
            pool_stride=2,
            dropout=0.1
        )
        
        # Block 3: 64 → 128
        self.block3 = ConvBlock1D(
            in_channels=64,
            out_channels=128,
            kernel_size=8,
            pool_size=4,
            pool_stride=2,
            dropout=0.1
        )
        
        # Block 4: 128 → 256
        self.block4 = ConvBlock1D(
            in_channels=128,
            out_channels=256,
            kernel_size=8,
            pool_size=4,
            pool_stride=2,
            dropout=0.1
        )
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 投影到输出维度
        self.fc = nn.Linear(256, output_dim)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (batch, 512) burst序列
            
        Returns:
            embedding: (batch, 512) 嵌入特征
        """
        # Reshape: (batch, 512) → (batch, 1, 512)
        x = x.unsqueeze(1)
        
        # 4个双卷积块
        x = self.block1(x)  # (batch, 32, 256)
        x = self.block2(x)  # (batch, 64, 128)
        x = self.block3(x)  # (batch, 128, 64)
        x = self.block4(x)  # (batch, 256, 32)
        
        # 全局平均池化
        x = self.global_pool(x)  # (batch, 256, 1)
        x = x.squeeze(-1)  # (batch, 256)
        
        # 投影到输出维度
        x = self.fc(x)  # (batch, 512)
        
        return x


class HybridEncoder(nn.Module):
    """
    混合编码器：分类器特征（30%）+ 独立编码器（70%）
    
    Args:
        classifier: 预训练的分类器（冻结）
        input_dim: 输入维度（默认512）
        output_dim: 输出维度（默认512）
        clf_weight: 分类器特征权重（默认0.3）
    """
    
    def __init__(self, classifier, input_dim=512, output_dim=512, clf_weight=0.3):
        super().__init__()
        
        # 分类器（冻结）
        self.classifier = classifier
        for param in self.classifier.parameters():
            param.requires_grad = False
        self.classifier.eval()
        
        # 独立编码器（可学习）
        self.independent_encoder = IndependentEncoder(
            input_dim=input_dim,
            output_dim=output_dim
        )
        
        # 权重
        self.clf_weight = clf_weight
        self.ind_weight = 1 - clf_weight
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
        
        # Xavier初始化融合层
        nn.init.xavier_uniform_(self.fusion[0].weight)
        nn.init.zeros_(self.fusion[0].bias)
        
        # 统计参数量
        ind_params = sum(p.numel() for p in self.independent_encoder.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        total_params = ind_params + fusion_params
        
        print(f"\nHybridEncoder initialized:")
        print(f"  - Classifier weight: {self.clf_weight:.1%}")
        print(f"  - Independent encoder weight: {self.ind_weight:.1%}")
        print(f"  - Independent encoder params: {ind_params/1000:.1f}K")
        print(f"  - Fusion layer params: {fusion_params/1000:.1f}K")
        print(f"  - Total trainable params: {total_params/1000:.1f}K")
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (batch, 512) burst序列
            
        Returns:
            embedding: (batch, 512) 混合嵌入特征
        """
        # 分类器特征（冻结，权重30%）
        with torch.no_grad():
            clf_feat = self.classifier.get_penultimate_layer(x)  # (batch, 512)
        clf_feat_weighted = clf_feat * self.clf_weight
        
        # 独立编码器特征（可学习，权重70%）
        ind_feat = self.independent_encoder(x)  # (batch, 512)
        ind_feat_weighted = ind_feat * self.ind_weight
        
        # 融合
        combined = torch.cat([clf_feat_weighted, ind_feat_weighted], dim=1)  # (batch, 1024)
        fused = self.fusion(combined)  # (batch, 512)
        
        return fused


# ========== 测试代码 ==========
if __name__ == '__main__':
    """测试 HybridEncoder"""
    
    # 创建模拟分类器
    class MockClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(512, 512)
        
        def get_penultimate_layer(self, x):
            return self.fc(x)
    
    classifier = MockClassifier()
    
    # 创建混合编码器
    encoder = HybridEncoder(
        classifier=classifier,
        input_dim=512,
        output_dim=512,
        clf_weight=0.3
    )
    
    # 测试前向传播
    batch_size = 16
    x = torch.randn(batch_size, 512)
    
    embedding = encoder(x)
    
    print(f"\n✓ Test passed!")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {embedding.shape}")
    
    # 测试梯度流
    loss = embedding.sum()
    loss.backward()
    
    has_grad = any(p.grad is not None for p in encoder.independent_encoder.parameters())
    clf_has_grad = any(p.grad is not None for p in classifier.parameters())
    
    print(f"\n✓ Gradient flow test:")
    print(f"  - Independent encoder has gradient: {has_grad}")
    print(f"  - Classifier has NO gradient: {not clf_has_grad}")
