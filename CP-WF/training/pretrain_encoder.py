"""
阶段1：预训练encoder

使用纯净流量 + 数据增强 + SupCon损失
"""

import sys
sys.path.append('..')
sys.path.append('../..')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import os
from tqdm import tqdm

# 导入自定义模块
from src.encoder import IndependentEncoder
from src.losses import SupConLoss
from src.augmentation import TrafficAugmentor


def load_clean_data(dataset='A', data_path='../../processed_data', use_full=False):
    """
    加载纯净流量数据
    
    Args:
        dataset: 数据集名称（A/B/C）
        data_path: 数据路径
        use_full: 是否使用完整数据集（True）或分类数据集（False）
        
    Returns:
        data: (N, 512) 纯净流量
        labels: (N,) 类别标签
    """
    print(f"\n加载纯净流量数据...")
    print(f"  Dataset: {dataset}")
    print(f"  Type: {'完整数据集' if use_full else '分类数据集'}")
    
    # 加载数据
    if use_full:
        # 使用完整数据集（train + val + test）
        train_file = os.path.join(data_path, 'cw100_train.npz')
        val_file = os.path.join(data_path, 'cw100_val.npz')
        test_file = os.path.join(data_path, 'cw100_test.npz')
        
        # 加载并合并
        train_dict = np.load(train_file)
        val_dict = np.load(val_file)
        test_dict = np.load(test_file)
        
        data = np.concatenate([
            train_dict['burst_sequences'],
            val_dict['burst_sequences'],
            test_dict['burst_sequences']
        ], axis=0)
        
        labels = np.concatenate([
            train_dict['labels'],
            val_dict['labels'],
            test_dict['labels']
        ], axis=0)
        
        print(f"  Files: cw100_train.npz + cw100_val.npz + cw100_test.npz")
    else:
        # 使用分类数据集（splited_data中的A/B/C）
        train_file = os.path.join(data_path, 'splited_data', f'cw100_train_{dataset}.npz')
        val_file = os.path.join(data_path, 'splited_data', f'cw100_val_{dataset}.npz')
        test_file = os.path.join(data_path, 'splited_data', f'cw100_test_{dataset}.npz')
        
        # 加载并合并
        train_dict = np.load(train_file)
        val_dict = np.load(val_file)
        test_dict = np.load(test_file)
        
        data = np.concatenate([
            train_dict['burst_sequences'],
            val_dict['burst_sequences'],
            test_dict['burst_sequences']
        ], axis=0)
        
        labels = np.concatenate([
            train_dict['labels'],
            val_dict['labels'],
            test_dict['labels']
        ], axis=0)
        
        print(f"  Files: cw100_train_{dataset}.npz + cw100_val_{dataset}.npz + cw100_test_{dataset}.npz")
    
    print(f"  Data shape: {data.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Num classes: {len(np.unique(labels))}")
    
    return data, labels


def train_one_epoch(encoder, criterion, optimizer, train_loader, augmentor, device):
    """
    训练一个epoch
    
    Args:
        encoder: 编码器
        criterion: 损失函数（SupConLoss）
        optimizer: 优化器
        train_loader: 训练数据加载器
        augmentor: 数据增强器
        device: 设备
        
    Returns:
        avg_loss: 平均损失
    """
    encoder.train()
    
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for data, labels in pbar:
        data = data.to(device)
        labels = labels.to(device)
        
        # 数据增强（生成两个视图）
        data_aug = augmentor.augment(data)
        
        # 提取特征
        feat1 = encoder(data.squeeze())  # (batch, 512)
        feat2 = encoder(data_aug.squeeze())  # (batch, 512)
        
        # 计算SupCon损失
        loss = criterion(feat1, feat2, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # 更新进度条
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def main(args):
    """主函数"""
    
    print("=" * 80)
    print("阶段1：预训练encoder")
    print("=" * 80)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # 加载数据
    data, labels = load_clean_data(args.dataset, args.data_path, args.use_full)
    
    # 转换为tensor
    data_tensor = torch.FloatTensor(data).unsqueeze(1).unsqueeze(1)  # (N, 1, 1, 512)
    labels_tensor = torch.LongTensor(labels)
    
    # 创建数据集和数据加载器
    dataset = TensorDataset(data_tensor, labels_tensor)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"\nDataLoader created:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num batches: {len(train_loader)}")
    
    # 创建encoder（只用独立编码器，不需要分类器）
    print(f"\n创建encoder...")
    print(f"  使用独立编码器（不依赖分类器）")
    
    from src.encoder import IndependentEncoder
    encoder = IndependentEncoder(
        input_dim=512,
        output_dim=512
    ).to(device)
    
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Total parameters: {num_params / 1e6:.1f}M")
    
    # 创建损失函数
    criterion = SupConLoss(temperature=args.temperature)
    
    # 创建优化器
    optimizer = optim.Adam(
        encoder.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 创建学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )
    
    # 创建数据增强器
    augmentor = TrafficAugmentor(
        methods=['random_swap'],
        swap_ratio=args.swap_ratio
    )
    
    # 训练
    print(f"\n开始训练...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Swap ratio: {args.swap_ratio}")
    print("-" * 80)
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        # 训练一个epoch
        avg_loss = train_one_epoch(
            encoder, criterion, optimizer, train_loader, augmentor, device
        )
        
        # 更新学习率
        scheduler.step()
        
        # 打印信息
        print(f"Epoch {epoch+1}/{args.epochs}: Loss = {avg_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.6f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            if args.use_full:
                # 完整数据集：直接使用，不加后缀
                save_path = os.path.join(args.save_dir, 'pretrained_encoder.pth')
            else:
                # 分类数据集：加数据集后缀
                save_path = os.path.join(args.save_dir, f'pretrained_encoder_{args.dataset}.pth')
            torch.save(encoder.state_dict(), save_path)
            print(f"  → Best model saved: {save_path}")
    
    print("-" * 80)
    print(f"\n✓ 预训练完成!")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Model saved: {save_path}")
    if args.use_full:
        print(f"\n注意：使用完整数据集训练，encoder可用于所有数据集")
    else:
        print(f"\n注意：使用数据集{args.dataset}训练，encoder适用于该数据集")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='预训练encoder')
    
    # 数据参数
    parser.add_argument('--dataset', type=str, default='A', choices=['A', 'B', 'C'],
                        help='数据集名称')
    parser.add_argument('--use_full', action='store_true',
                        help='使用完整数据集（默认使用分类数据集cw100）')
    parser.add_argument('--data_path', type=str, default='../../processed_data',
                        help='数据路径')
    parser.add_argument('--model_path', type=str, default='../../saved_models',
                        help='分类器模型路径')
    
    # 模型参数
    parser.add_argument('--classifier', type=str, default='WFModel',
                        choices=['WFModel', 'DF', 'LSTMClassifier', 'TMWF'],
                        help='分类器名称（仅用于阶段2训练denoiser时创建混合编码器）')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=30,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='批大小')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='SupCon温度参数')
    
    # 数据增强参数
    parser.add_argument('--swap_ratio', type=float, default=0.05,
                        help='随机交换比例')
    
    # 保存参数
    parser.add_argument('--save_dir', type=str, default='../saved_models',
                        help='模型保存路径')
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 运行
    main(args)
