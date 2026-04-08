"""
评估脚本 - 简洁易用版本

功能：
1. 评估净化质量（MSE）
2. 评估分类准确率（原始 vs 对抗 vs 净化）
3. 支持跨分类器测试
4. 自动保存结果

用法示例：
# 基础用法（使用默认路径）
python evaluate.py --model improved_WT_Full_best.pth --defense WT

# 指定测试分类器
python evaluate.py --model improved_WT_Full_best.pth --defense WT --classifier DF

# 使用子数据集
python evaluate.py --model improved_WT_A_best.pth --defense WT --dataset A

# 保存结果
python evaluate.py --model improved_WT_Full_best.pth --defense WT --save
"""

import sys
import os

# 添加路径
script_dir = os.path.dirname(os.path.abspath(__file__))
improved_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(improved_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, improved_dir)

import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
import json
from datetime import datetime

# 导入模块
from model.classifier import WFModel
from model.DF import DF
from model.LSTM_classifier import LSTMClassifier
from model.TMWF import TMWF
from refined_denoiser.src.encoder import IndependentEncoder
from refined_denoiser.src.denoiser import TransUNet
from refined_denoiser.src.data_loader import get_adversarial_dataloader


def parse_args():
    parser = argparse.ArgumentParser(
        description='评估改进的流量净化器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 基础评估（完整数据集）
  python evaluate.py --model improved_WT_Full_best.pth --defense WT
  
  # 使用不同的测试分类器
  python evaluate.py --model improved_WT_Full_best.pth --defense WT --classifier DF
  
  # 评估子数据集
  python evaluate.py --model improved_WT_A_best.pth --defense WT --dataset A
  
  # 保存结果
  python evaluate.py --model improved_WT_Full_best.pth --defense WT --save
  
  # 完整参数
  python evaluate.py \\
      --model improved_WT_Full_best.pth \\
      --defense WT \\
      --classifier DF \\
      --batch_size 64 \\
      --save
        """
    )
    
    # 必需参数
    parser.add_argument('--model', type=str, required=True,
                        help='训练好的denoiser模型文件名（在models/目录下）')
    parser.add_argument('--defense', type=str, required=True,
                        choices=['WT', 'adv'],
                        help='防御方法：WT 或 adv')
    
    # 可选参数
    parser.add_argument('--classifier', type=str, default='WFModel',
                        choices=['WFModel', 'DF', 'LSTMClassifier', 'TMWF'],
                        help='测试分类器（默认：WFModel）')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['A', 'B', 'C'],
                        help='数据集（A/B/C），不指定则使用完整数据集')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批大小（默认：32）')
    
    # 路径参数（通常不需要修改）
    parser.add_argument('--model_dir', type=str, default='improved_denoiser/models',
                        help='Denoiser模型目录')
    parser.add_argument('--encoder_path', type=str,
                        default='refined_denoiser/saved_models/pretrained_encoder.pth',
                        help='预训练encoder路径')
    parser.add_argument('--classifier_dir', type=str, default='saved_models',
                        help='分类器模型目录')
    parser.add_argument('--data_path', type=str, default='processed_data',
                        help='数据路径')
    
    # 保存参数
    parser.add_argument('--save', action='store_true',
                        help='保存评估结果到JSON文件')
    parser.add_argument('--result_dir', type=str, default='improved_denoiser/results',
                        help='结果保存目录')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='设备选择（默认：cuda:0）')
    
    return parser.parse_args()


def load_classifier(classifier_name, num_classes, classifier_dir, dataset=None, device='cuda'):
    """加载分类器"""
    # 构建模型路径
    if dataset is None:
        # 完整数据集
        model_path = os.path.join(classifier_dir, f'{classifier_name}.pth')
    else:
        # 子数据集
        model_path = os.path.join(classifier_dir, f'{classifier_name}_{dataset}.pth')
    
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
    classifier.load_state_dict(torch.load(model_path, map_location=device))
    classifier.to(device)
    classifier.eval()
    
    return classifier, model_path


def evaluate(args):
    """主评估函数"""
    
    # 确定数据集类型
    use_full = (args.dataset is None)
    dataset_name = 'Full' if use_full else args.dataset
    num_classes = 100  # 都是100类
    
    print("\n" + "=" * 80)
    print("评估改进的流量净化器")
    print("=" * 80)
    print(f"模型: {args.model}")
    print(f"防御方法: {args.defense}")
    print(f"数据集: {'完整数据集（100类）' if use_full else f'数据集{args.dataset}（100类）'}")
    print(f"测试分类器: {args.classifier}")
    print("=" * 80 + "\n")
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}\n")
    
    # 1. 加载encoder
    print("加载模型...")
    encoder = IndependentEncoder(input_dim=512, output_dim=512).to(device)
    if os.path.exists(args.encoder_path):
        encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))
        print(f"✓ Encoder: {args.encoder_path}")
    else:
        print(f"⚠ Encoder not found: {args.encoder_path}")
        print(f"  Using random initialization")
    encoder.eval()
    
    # 2. 加载denoiser
    denoiser_path = os.path.join(args.model_dir, args.model)
    if not os.path.exists(denoiser_path):
        print(f"\n❌ 错误：找不到模型文件 {denoiser_path}")
        print(f"\n可用的模型文件：")
        if os.path.exists(args.model_dir):
            models = [f for f in os.listdir(args.model_dir) if f.endswith('.pth')]
            for m in models:
                print(f"  - {m}")
        else:
            print(f"  模型目录不存在: {args.model_dir}")
        return
    
    denoiser = TransUNet(img_ch=1, output_ch=1).to(device)
    denoiser.load_state_dict(torch.load(denoiser_path, map_location=device))
    denoiser.eval()
    print(f"✓ Denoiser: {denoiser_path}")
    
    # 3. 加载分类器
    classifier, clf_path = load_classifier(
        args.classifier,
        num_classes,
        args.classifier_dir,
        args.dataset,
        device
    )
    print(f"✓ Classifier: {clf_path}\n")
    
    # 4. 加载测试数据
    print("加载测试数据...")
    test_loader = get_adversarial_dataloader(
        datasets=None if use_full else [args.dataset],
        defense=args.defense,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        data_path=args.data_path,
        use_full=use_full
    )
    print(f"✓ 测试数据: {len(test_loader)} batches (~{len(test_loader) * args.batch_size} samples)\n")
    
    # 5. 评估
    print("开始评估...")
    print("-" * 80)
    
    total_mse = 0
    correct_original = 0
    correct_adversarial = 0
    correct_denoised = 0
    total_samples = 0
    
    with torch.no_grad():
        for adv_data, clean_data, labels in tqdm(test_loader, desc='Evaluating'):
            adv_data = adv_data.to(device)
            clean_data = clean_data.to(device)
            labels = labels.to(device)
            
            batch_size = adv_data.shape[0]
            
            # 净化
            denoised_data = denoiser(adv_data)
            
            # 计算MSE
            mse = torch.mean((denoised_data - clean_data) ** 2).item()
            total_mse += mse * batch_size
            
            # 准备分类器输入：(batch, 1, 1, 512) -> (batch, 512)
            clean_seq = clean_data.squeeze()
            adv_seq = adv_data.squeeze()
            denoised_seq = denoised_data.squeeze()
            
            # 分类
            pred_original = classifier(clean_seq).argmax(dim=1)
            pred_adversarial = classifier(adv_seq).argmax(dim=1)
            pred_denoised = classifier(denoised_seq).argmax(dim=1)
            
            # 统计准确率
            correct_original += (pred_original == labels).sum().item()
            correct_adversarial += (pred_adversarial == labels).sum().item()
            correct_denoised += (pred_denoised == labels).sum().item()
            total_samples += batch_size
    
    # 6. 计算指标
    avg_mse = total_mse / total_samples
    acc_original = 100.0 * correct_original / total_samples
    acc_adversarial = 100.0 * correct_adversarial / total_samples
    acc_denoised = 100.0 * correct_denoised / total_samples
    
    improvement_vs_adv = acc_denoised - acc_adversarial
    improvement_vs_orig = acc_denoised - acc_original
    
    # 7. 打印结果
    print("-" * 80)
    print("\n📊 评估结果")
    print("=" * 80)
    
    print("\n净化质量:")
    print(f"  MSE (净化 vs 原始): {avg_mse:.6f}")
    
    print(f"\n分类准确率:")
    print(f"  原始流量:   {acc_original:6.2f}%")
    print(f"  对抗流量:   {acc_adversarial:6.2f}%  ← 攻击效果")
    print(f"  净化流量:   {acc_denoised:6.2f}%  ← 防御效果")
    
    print(f"\n改进效果:")
    print(f"  vs 对抗流量: {improvement_vs_adv:+6.2f}%  ← 净化提升")
    print(f"  vs 原始流量: {improvement_vs_orig:+6.2f}%  ← 与原始差距")
    
    print(f"\n测试配置:")
    print(f"  测试分类器: {args.classifier}")
    print(f"  防御方法: {args.defense}")
    print(f"  数据集: {dataset_name}")
    
    print("\n" + "=" * 80)
    print("✓ 评估完成!")
    print("=" * 80 + "\n")
    
    # 8. 保存结果
    if args.save:
        results = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model': args.model,
            'defense': args.defense,
            'dataset': dataset_name,
            'test_classifier': args.classifier,
            'metrics': {
                'mse': float(avg_mse),
                'acc_original': float(acc_original),
                'acc_adversarial': float(acc_adversarial),
                'acc_denoised': float(acc_denoised),
                'improvement_vs_adv': float(improvement_vs_adv),
                'improvement_vs_orig': float(improvement_vs_orig)
            },
            'config': {
                'batch_size': args.batch_size,
                'num_samples': total_samples,
                'device': str(device)
            }
        }
        
        # 生成结果文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(
            args.result_dir,
            f'eval_{args.classifier}_{dataset_name}_{args.defense}_{timestamp}.json'
        )
        
        os.makedirs(args.result_dir, exist_ok=True)
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"结果已保存到: {result_file}\n")
    
    return results if args.save else None


def main():
    args = parse_args()
    
    try:
        evaluate(args)
    except KeyboardInterrupt:
        print("\n\n⚠ 评估被用户中断")
    except Exception as e:
        print(f"\n\n❌ 评估出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
