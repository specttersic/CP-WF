import torch
from torch import nn
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, pool_size, pool_stride, dropout_p, activation):
        super(ConvBlock, self).__init__()
        padding = kernel_size // 2 
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            activation(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            activation(inplace=True),
            nn.MaxPool1d(pool_size, pool_stride, padding=0),
            nn.Dropout(p=dropout_p)
        )

    def forward(self, x):
        return self.block(x)

class DF(nn.Module):
    def __init__(self, num_classes):
        super(DF, self).__init__()
        
        filter_num = [32, 64, 128, 256]
        kernel_size = 8
        conv_stride_size = 1
        pool_stride_size = 4
        pool_size = 8
        length_after_extraction = 1
        
        self.feature_extraction = nn.Sequential(
            ConvBlock(1, filter_num[0], kernel_size, conv_stride_size, pool_size, pool_stride_size, 0.1, nn.ELU),
            ConvBlock(filter_num[0], filter_num[1], kernel_size, conv_stride_size, pool_size, pool_stride_size, 0.1, nn.ReLU),
            ConvBlock(filter_num[1], filter_num[2], kernel_size, conv_stride_size, pool_size, pool_stride_size, 0.1, nn.ReLU),
            ConvBlock(filter_num[2], filter_num[3], kernel_size, conv_stride_size, pool_size, pool_stride_size, 0.1, nn.ReLU)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filter_num[3] * length_after_extraction, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # ==================================================================
        # 【新增代码】 自动适配输入维度
        # ==================================================================
        
        # 1. 确保数据是 float 类型 (解决 double/float 不匹配问题)
        if x.dtype != torch.float32:
            x = x.float()

        # 2. 解决维度问题
        # 情况 A: 输入形状为 [1, 128, 512] (你的报错情况)
        # 目标: [128, 1, 512] (Batch=128, Channel=1, Length=512)
        if x.dim() == 3 and x.shape[0] == 1 and x.shape[1] > 1:
            # 强制重塑：忽略第0维，将 Batch 放第一维，中间插入 Channel=1
            x = x.view(-1, 1, x.shape[-1])
            
        # 情况 B: 输入形状为 [128, 512] (普通二维数据)
        # 目标: [128, 1, 512]
        elif x.dim() == 2:
            x = x.unsqueeze(1)
            
        # 情况 C: 输入形状为 [128, 1, 512] (已经是正确的) -> 不做处理
        # ==================================================================

        x = self.feature_extraction(x)
        x = self.classifier(x)
        
        return x
