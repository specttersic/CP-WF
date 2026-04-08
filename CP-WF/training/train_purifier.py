"""
统一训练脚本 - 改进的流量净化器

损失函数：MSE + Perceptual Loss + SupCon

用法：
# 完整数据集训练
python train_improved.py --defense all --use_full

# 单数据集训练（用于跨数据集测试）
python train_improved.py --defense all --datasets A

# 多数据集组合训练
python train_improved.py --defense mockingbird --datasets A B
"""

import sys
import os

# 添加路径
script_dir = os.path.dirname(os.path.abspath(__file__))
improved_dir = os.path.dirname(script_dir)  # improved_denoiser/
project_root = os.path.dirname(improved_dir)  # traffic_denoise/
sys.path.insert(0, project_root)
sys.path.insert(0, improved_dir)

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from tqdm import tqdm
import json
from datetime import datetime

# 导入模块
from model.classifier import WFModel
from model.DF import DF
from model.LSTM_classifier import LSTMClassifier
from model.TMWF import TMWF
from model.VarCNN import VarCNN
from refined_denoiser.src.encoder import IndependentEncoder
from refined_denoiser.src.denoiser import TransUNet
from refined_denoiser.src.data_loader import get_adversarial_dataloader
from src.improved_losses import ImprovedTripleLoss


def parse_args():
    parser = argparse.ArgumentParser(description='训练改进的流量净化器')
    
    # 数据参数
    parser.add_argument('--defense', type=str, nargs='+', default=['mockingbird'],
                        help='防御方法：mockingbird, WT, adv 或 all（可指定多个）')
    parser.add_argument('--use_full', action='store_true',
                        help='使用完整数据集（A+B+C合并）')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='数据集列表：A, B, C 或 all（可指定多个，如：--datasets A B）')
    parser.add_argument('--data_path', type=str, 
                        default='../../processed_data',
                        help='数据路径')
    
    # 模型参数
    parser.add_argument('--encoder_path', type=str,
                        default='../saved_models/pretrained_encoder.pth',
                        help='预训练encoder路径')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=40,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    
    # 损失权重
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='MSE损失的权重（默认0.5）')
    parser.add_argument('--beta', type=float, default=0.25,
                        help='Perceptual Loss的权重（默认0.25）')
    parser.add_argument('--gamma', type=float, default=0.25,
                        help='SupCon损失的权重（默认0.25）')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='对比学习温度（默认0.5）')
    
    # 保存路径
    parser.add_argument('--save_dir', type=str, default='../saved_models',
                        help='模型保存路径')
    parser.add_argument('--result_dir', type=str, default='../results',
                        help='结果保存路径')
    
    # 设备
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='设备选择')
    
    return parser.parse_args()


def load_classifier(classifier_name, dataset, num_classes, device, classifier_path=None, use_full=False):
    """加载分类器"""
    # 如果没有指定路径，自动推断
    if classifier_path is None:
        if use_full:
            # 完整数据集：不带后缀
            classifier_path = f'saved_models/{classifier_name}.pth'
        else:
            # 子数据集：在split文件夹下
            classifier_path = f'saved_models/split/{classifier_name}_{dataset}.pth'
    
    # 创建分类器
    if classifier_name == 'WFModel':
        classifier = WFModel(input_dim=512, num_classes=num_classes, hidden_dim=512)
    elif classifier_name == 'DF':
        classifier = DF(num_classes=num_classes)
    elif classifier_name == 'LSTMClassifier':
        classifier = LSTMClassifier(
            input_size=1, hidden_size=128, num_layers=2,
            num_classes=num_classes, dropout=0.5
        )
    elif classifier_name == 'TMWF':
        classifier = TMWF(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")
    
    # 加载权重
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier.to(device)
    classifier.eval()
    
    # 冻结参数
    for param in classifier.parameters():
        param.requires_grad = False
    
    return classifier


def train_one_epoch(denoiser, encoder, train_loader, 
                   criterion, optimizer, device):
    """训练一个epoch"""
    denoiser.train()
    encoder.eval()
    
    total_loss = 0
    loss_details = {
        'mse': 0,
        'perceptual': 0,
        'supcon': 0
    }
    
    pbar = tqdm(train_loader, desc='Training', leave=False)
    for adv_data, clean_data, labels in pbar:
        adv_data = adv_data.to(device)
        clean_data = clean_data.to(device)
        labels = labels.to(device)
        
        # 前向传播
        purified_data = denoiser(adv_data)
        
        # 计算损失（不需要classifier参数）
        loss, loss_dict = criterion(purified_data, clean_data, labels, encoder)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 记录
        total_loss += loss.item()
        for key in loss_details:
            if key in loss_dict:
                loss_details[key] += loss_dict[key]
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mse': f'{loss_dict["mse"]:.4f}'
        })
    
    n_batches = len(train_loader)
    avg_loss = total_loss / n_batches
    avg_details = {k: v / n_batches for k, v in loss_details.items()}
    
    return avg_loss, avg_details


def train_denoiser(defense, args, is_splited=False):
    """训练单个防御方法的净化器"""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 确定数据集描述
    if args.use_full:
        dataset_desc = "完整数据集（A+B+C）"
        datasets = None
        use_full = True
    else:
        dataset_desc = f"数据集 {'+'.join(args.datasets)}"
        datasets = args.datasets
        use_full = False
    
    print(f"\n{'='*80}")
    print(f"训练改进的净化器 - {defense}")
    print(f"{'='*80}")
    print(f"损失配置: α={args.alpha}, β={args.beta}, γ={args.gamma}")
    print(f"数据集: {dataset_desc}")
    print(f"设备: {device}")
    print(f"{'='*80}\n")
    
    # 1. 加载encoder
    print("加载模型...")
    encoder = IndependentEncoder(input_dim=512, output_dim=512).to(device)
    if os.path.exists(args.encoder_path):
        encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))
        print(f"✓ Encoder loaded: {args.encoder_path}")
    else:
        print(f"⚠ Encoder not found: {args.encoder_path}")
        print(f"  Using random initialization")
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    
    # 2. 创建denoiser
    denoiser = TransUNet(img_ch=1, output_ch=1).to(device)
    print(f"✓ Denoiser created")
    
    # 3. 加载数据
    print(f"加载数据...")
    train_loader = get_adversarial_dataloader(
        datasets=datasets,
        defense=defense,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        data_path=args.data_path,
        use_full=use_full
    )
    print(f"✓ Data loaded: {len(train_loader)} batches\n")
    
    # 4. 创建损失函数
    print("创建损失函数...\n")
    criterion = ImprovedTripleLoss(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        temperature=args.temperature
    )
    
    # 5. 优化器和调度器
    optimizer = Adam(denoiser.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 6. 训练
    print(f"开始训练...\n")
    best_loss = float('inf')
    history = []
    
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        
        # 训练
        avg_loss, loss_details = train_one_epoch(
            denoiser, encoder, train_loader,
            criterion, optimizer, device
        )
        
        # 记录
        metrics = {
            'epoch': epoch,
            'lr': optimizer.param_groups[0]['lr'],
            'total_loss': avg_loss,
            **loss_details
        }
        history.append(metrics)
        
        # 打印
        print(f"  Loss: {avg_loss:.4f}")
        print(f"    - MSE: {loss_details['mse']:.4f}")
        print(f"    - Perceptual: {loss_details['perceptual']:.4f}")
        print(f"    - SupCon: {loss_details['supcon']:.4f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            
            # 根据数据集类型确定模型名称和保存路径
            if args.use_full:
                model_name = f"denoiser_{defense}.pth"
                save_dir = args.save_dir
            elif is_splited:
                # 单个数据集训练，保存到splited目录
                model_name = f"denoiser_{defense}_dataset{''.join(args.datasets)}.pth"
                save_dir = os.path.join(args.save_dir, 'splited')
            else:
                # 多个数据集组合训练
                model_name = f"denoiser_{defense}_{'_'.join(args.datasets)}.pth"
                save_dir = args.save_dir
            
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, model_name)
            
            torch.save(denoiser.state_dict(), model_path)
            print(f"  ✓ Best model saved: {model_path}")
        
        scheduler.step()
        print()
    
    # 7. 保存训练历史
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = "完整数据集" if args.use_full else "_".join(args.datasets)
    
    # 根据数据集类型确定结果保存路径
    if is_splited:
        result_dir = os.path.join(args.result_dir, 'splited')
    else:
        result_dir = args.result_dir
    
    os.makedirs(result_dir, exist_ok=True)
    history_file = os.path.join(
        result_dir,
        f'training_history_{defense}_{dataset_name}_{timestamp}.json'
    )
    
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump({
            'config': vars(args),
            'defense': defense,
            'dataset_info': {
                'use_full': args.use_full,
                'datasets': args.datasets if not args.use_full else ['A', 'B', 'C'],
                'dataset_name': dataset_name
            },
            'method': 'improved_multi_scale_mse',
            'best_loss': best_loss,
            'model_path': model_path,
            'history': history
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Training history saved: {history_file}")
    print(f"✓ Best loss: {best_loss:.4f}\n")
    
    return model_path, best_loss


def main():
    args = parse_args()
    
    # 处理数据集参数
    if args.datasets and 'all' in args.datasets:
        args.datasets = ['A', 'B', 'C']
    elif not args.use_full and not args.datasets:
        args.datasets = ['A']  # 默认使用数据集A
    
    # 判断是否为单数据集训练（用于跨数据集测试）
    is_splited = (not args.use_full and args.datasets and len(args.datasets) == 1)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    
    # 确定要训练的防御方法
    if 'all' in args.defense:
        defenses = ['mockingbird', 'WT', 'adv']
    else:
        defenses = args.defense
    
    # 打印配置信息
    print(f"\n{'='*80}")
    print(f"训练配置")
    print(f"{'='*80}")
    print(f"方法: 改进的损失函数（MSE + Perceptual + SupCon）")
    print(f"防御方法: {', '.join(defenses)}")
    if args.use_full:
        print(f"数据集: 完整数据集（A+B+C）")
        print(f"模型保存: {args.save_dir}/")
    elif is_splited:
        print(f"数据集: 单数据集 {args.datasets[0]}（跨数据集训练）")
        print(f"模型保存: {args.save_dir}/splited/")
    else:
        print(f"数据集: {'+'.join(args.datasets)}")
        print(f"模型保存: {args.save_dir}/")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"损失权重: α={args.alpha}, β={args.beta}, γ={args.gamma}")
    print(f"{'='*80}\n")
    
    # 训练
    if len(defenses) > 1 or (is_splited and len(args.datasets) == 1):
        results = {}
        total = len(defenses) * (len(args.datasets) if is_splited else 1)
        current = 0
        
        for defense in defenses:
            if is_splited:
                # 单数据集训练模式
                for dataset in args.datasets:
                    current += 1
                    print(f"\n[{current}/{total}] 训练 {defense} - 数据集{dataset}")
                    args.datasets = [dataset]
                    model_path, best_loss = train_denoiser(defense, args, is_splited=True)
                    results[f"{defense}_{dataset}"] = {
                        'model_path': model_path,
                        'best_loss': best_loss
                    }
            else:
                current += 1
                print(f"\n[{current}/{total}] 训练 {defense}")
                model_path, best_loss = train_denoiser(defense, args, is_splited=False)
                results[defense] = {
                    'model_path': model_path,
                    'best_loss': best_loss
                }
        
        # 打印总结
        print(f"\n{'='*80}")
        print(f"训练完成总结")
        print(f"{'='*80}\n")
        for key, result in results.items():
            print(f"{key:20s}: {result['best_loss']:.4f} - {result['model_path']}")
        print()
    else:
        train_denoiser(defenses[0], args, is_splited=is_splited)


if __name__ == '__main__':
    main()
