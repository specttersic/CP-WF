import torch
import torch.nn as nn
import torch.nn.functional as F

"""
TransUNet-1D for Adversarial Traffic Denoising

核心创新：
1. Attention Gate机制：自适应过滤跳跃连接中的防御噪声
2. Transformer Bottleneck：捕获长距离依赖


模型架构：
- 编码器：5层CNN，提取多尺度特征（64→128→256→512→1024）
- Bottleneck：Transformer，捕获全局依赖
- 解码器：4层上采样 + Attention Gates，逐步恢复分辨率并过滤噪声

Attention Gate工作原理：
- 输入：g（解码器特征，门控信号）+ x（编码器特征，跳跃连接）
- 输出：过滤后的编码器特征 x_filtered = x * attention_weights
- 作用：解码器特征指导编码器特征的过滤，抑制防御噪声（如dummy packets）

为什么有效：
- 解码器在上采样过程中逐步恢复原始流量，其特征包含"什么是干净的"信息
- 用这个信息作为门控信号，自适应地过滤编码器特征中的噪声
- 多层级过滤：浅层过滤低级噪声，深层过滤高级噪声
"""

# ========== 基础卷积模块 (保持不变) ==========
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2)),
            nn.Conv2d(ch_in, ch_out, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)

# ========== [创新点 1] 新增 Attention Gate 模块 ==========
# 作用：在跳跃连接中过滤噪声（Padding/Dummy Packets），只保留有效特征
# 
# 工作原理：
# 1. 解码器特征g作为"门控信号"（已部分净化，知道什么是干净的）
# 2. 编码器特征x作为"待过滤特征"（含噪声，需要选择性保留）
# 3. 生成注意力权重psi ∈ [0,1]，高权重=有效特征，低权重=噪声
# 4. 输出：x_filtered = x * psi（自适应过滤）
#
# 为什么比直接拼接好：
# - 标准UNet直接拼接会传递防御噪声（如WT的dummy packets）
# - AG能自适应地为每个位置生成不同的权重
# - 保留有效的流量模式，抑制防御噪声
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g: 门控信号的通道数 (来自Decoder)
            F_l: 跳跃连接特征的通道数 (来自Encoder)
            F_int: 中间层通道数
        """
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Args:
            g: 解码器特征（门控信号，已部分净化）- shape: (batch, F_g, H, W)
            x: 编码器特征（跳跃连接，含噪声）- shape: (batch, F_l, H, W)
        
        Returns:
            x_filtered: 过滤后的编码器特征 - shape: (batch, F_l, H, W)
        
        工作流程：
            1. 特征对齐：W_g(g) 和 W_x(x) 投影到同一空间（F_int维）
            2. 特征融合：relu(W_g(g) + W_x(x))，结合两个特征的信息
            3. 注意力生成：sigmoid(psi(融合特征))，生成[0,1]的权重
            4. 自适应过滤：x * attention_weights，保留有效部分
        """
        # g: Decoder feature (Gating signal)
        # x: Encoder feature (Skip connection)
        
        # 1. 对齐 g 和 x 的特征图尺寸 (通常 g 是 x 的一半，但这里经过 upsample 后 g 应该和 x 差不多)
        # 如果尺寸不一致，需要 interpolate g
        if g.shape[2:] != x.shape[2:]:
             g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)

        # 2. 特征变换：投影到同一空间
        g1 = self.W_g(g)  # 解码器特征投影 (batch, F_int, H, W)
        x1 = self.W_x(x)  # 编码器特征投影 (batch, F_int, H, W)
        
        # 3. 特征融合 + 注意力生成
        psi = self.relu(g1 + x1)  # 融合两个特征 (batch, F_int, H, W)
        psi = self.psi(psi)        # 生成注意力系数 map [0, 1] (batch, 1, H, W)

        # 4. 自适应过滤：高权重位置保留，低权重位置抑制
        return x * psi  # 过滤后的特征 (batch, F_l, H, W)

# ========== Transformer Bottleneck 模块 ==========
# 作用：捕获长距离依赖，增强全局理解能力
#
# 为什么需要Transformer：
# - CNN的感受野有限，难以捕获整个512长度序列的全局依赖
# - 流量序列的不同位置可能有重要的关联（如初始握手和后续传输）
# - Transformer的自注意力机制能关注任意位置的关系
#
# 工作原理：
# 1. MultiHeadAttention：计算序列中每个位置与其他位置的相关性
# 2. FeedForward：非线性变换，增强表达能力
# 3. 残差连接 + LayerNorm：稳定训练，保留原始信息
class TransformerBlock1D(nn.Module):
    def __init__(self, dim, num_heads=4, ff_mult=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.squeeze(2).transpose(1, 2)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x.transpose(1, 2).unsqueeze(2)

# ========== 改进后的 TransUNet-1D (集成 AG) ==========
# 
# 完整架构：
# 输入 (batch, 1, 1, 512) - 对抗流量
#   ↓
# 编码器（5层CNN，多尺度特征提取）
#   Conv1: 1→64   (1×512)
#   Conv2: 64→128 (1×256) ↓ MaxPool
#   Conv3: 128→256 (1×128) ↓ MaxPool
#   Conv4: 256→512 (1×64)  ↓ MaxPool
#   Conv5: 512→1024 (1×32) ↓ MaxPool
#   ↓
# Transformer Bottleneck（全局依赖建模）
#   MultiHeadAttention (1024-dim, 4 heads)
#   FeedForward (1024→4096→1024)
#   ↓
# 解码器（4层上采样 + Attention Gates）
#   Up5: 1024→512 (1×64)  ↑ Upsample + AG(x4) + Conv
#   Up4: 512→256  (1×128) ↑ Upsample + AG(x3) + Conv
#   Up3: 256→128  (1×256) ↑ Upsample + AG(x2) + Conv
#   Up2: 128→64   (1×512) ↑ Upsample + AG(x1) + Conv
#   ↓
# 输出 (batch, 1, 1, 512) - 净化流量

# 关键创新：
# 1. Attention Gates：在每个跳跃连接处过滤噪声
# 2. 多层级过滤：浅层过滤低级噪声，深层过滤高级噪声
# 3. Transformer：捕获全局依赖，理解整个序列的模式
class TransUNetRaw(nn.Module):
    def __init__(self, img_ch=1, output_ch=1,
                 num_heads=4, transformer_layers=1, dropout=0.1):
        super(TransUNetRaw, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        # 编码器
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        # Transformer Bottleneck
        blocks = []
        for _ in range(transformer_layers):
            blocks.append(TransformerBlock1D(dim=1024, num_heads=num_heads,
                                             dropout=dropout))
        self.transformer = nn.Sequential(*blocks)

        # 解码器 + [创新点] Attention Gates
        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256) # 新增 AG
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128) # 新增 AG
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)  # 新增 AG
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)    # 新增 AG
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # ---------- 编码路径 ----------
        x1 = self.Conv1(x)
        x2 = self.Conv2(self.Maxpool(x1))
        x3 = self.Conv3(self.Maxpool(x2))
        x4 = self.Conv4(self.Maxpool(x3))
        x5 = self.Conv5(self.Maxpool(x4))

        # ---------- Transformer Bottleneck ----------
        x5 = self.transformer(x5)

        # ---------- 解码路径 (集成 AG) ----------
        d5 = self.Up5(x5)
        # 创新：使用 AG 过滤 x4，而不是直接 cat
        x4 = self.Att5(g=d5, x=x4) 
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d0 = self.Conv_1x1(d2)
        return d0
