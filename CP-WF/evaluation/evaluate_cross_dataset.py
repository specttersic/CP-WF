"""
统一测试脚本 - 改进的流量净化器

支持：跨分类器测试、跨数据集测试

用法：
# 测试单个模型
python test_splited.py --defense mockingbird --datasets A

# 跨分类器测试
python test_splited.py --defense mockingbird --datasets A --cross_classifier

# 跨数据集测试
python test_splited.py --defense mockingbird --train_dataset A --cross_dataset

# 测试所有组合
python test_splited.py --test_all --cross_classifier
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
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime
import glob

# 导入模块
from model.classifier import WFModel
from model.DF import DF
from model.LSTM_classifier import LSTMClassifier
from model.TMWF import TMWF
from model.VarCNN import VarCNN
from refined_denoiser.src.encoder import IndependentEncoder
from refined_denoiser.src.denoiser import TransUNet
from refined_denoiser.src.data_loader import get_adversarial_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description='测试改进的流量净化器（子数据集）')
    
    # 测试模式
    parser.add_argument('--test_all', action='store_true',
                        help='测试所有防御方法和数据集的组合')
    parser.add_argument('--cross_classifier', action='store_true',
                        help='跨分类器测试（测试所有分类器）')
    parser.add_argument('--cross_dataset', action='store_true',
                        help='跨数据集测试（泛化能力）')
    
    # 数据参数
    parser.add_argument('--defense', type=str, default='mockingbird',
                        choices=['mockingbird', 'WT', 'adv'],
                        help='防御方法')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='测试数据集列表（如：A B C）')
    parser.add_argument('--train_dataset', type=str,
                        help='训练数据集（跨数据集测试时指定）')
    parser.add_argument('--use_full', action='store_true',
                        help='使用完整数据集测试')
    parser.add_argument('--data_path', type=str, default='../../processed_data',
                        help='数据路径')
    
    # 模型参数
    parser.add_argument('--classifier', type=str, default='VarCNN',
                        choices=['WFModel', 'DF', 'LSTMClassifier', 'TMWF', 'VarCNN'],
                        help='分类器类型（单分类器测试时使用）')
    parser.add_argument('--model_dir', type=str, default='../saved_models',
                        help='模型目录')
    parser.add_argument('--encoder_path', type=str,
                        default='../saved_models/pretrained_encoder.pth',
                        help='预训练encoder路径')
    parser.add_argument('--classifier_dir', type=str, default='../../saved_models',
                        help='分类器目录')
    
    # 测试参数
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批大小')
    
    # 保存路径
    parser.add_argument('--result_dir', type=str, default='../results',
                        help='结果保存路径')
    
    # 设备
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='设备选择')
    
    return parser.parse_args()


def get_model_path(defense, dataset, model_dir, use_full=False):
    """根据防御方法和数据集获取模型路径"""
    if use_full:
        model_name = f"denoiser_{defense}.pth"
        model_path = os.path.join(model_dir, model_name)
    else:
        model_name = f"denoiser_{defense}_dataset{dataset}.pth"
        model_path = os.path.join(model_dir, 'splited', model_name)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型不存在: {model_path}")
    
    return model_path


def load_classifier(classifier_name, num_classes, classifier_dir, dataset, device='cuda', use_full=False):
    """加载分类器"""
    # 根据是否使用完整数据集选择路径
    if use_full:
        # 完整数据集的分类器
        model_path = os.path.join(classifier_dir, f'{classifier_name}.pth')
        model_num_classes = num_classes
    else:
        # 子数据集的分类器在split文件夹下
        model_path = os.path.join(classifier_dir, 'split', f'{classifier_name}_{dataset}.pth')
        
        # 特殊处理：VarCNN的分类器是按照数据集实际类别数训练的
        # 其他分类器（DF、LSTM、TMWF）都是100类的
        if classifier_name == 'VarCNN':
            model_num_classes = num_classes  # 使用数据集的实际类别数
        else:
            model_num_classes = 100  # 其他分类器都是100类的
    
    # 创建分类器（使用模型实际的类别数）
    if classifier_name == 'WFModel':
        classifier = WFModel(input_dim=512, num_classes=model_num_classes, hidden_dim=512)
    elif classifier_name == 'DF':
        classifier = DF(num_classes=model_num_classes)
    elif classifier_name == 'LSTMClassifier':
        classifier = LSTMClassifier(
            input_size=1, hidden_size=128, num_layers=2,
            num_classes=model_num_classes, dropout=0.5
        )
    elif classifier_name == 'TMWF':
        classifier = TMWF(num_classes=model_num_classes)
    elif classifier_name == 'VarCNN':
        classifier = VarCNN(num_classes=model_num_classes)
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")
    
    classifier.load_state_dict(torch.load(model_path, map_location=device))
    classifier.to(device)
    classifier.eval()
    
    return classifier


def test_denoiser(denoiser, encoder, classifier, test_loader, device, is_varcnn=False):
    """测试净化器"""
    denoiser.eval()
    encoder.eval()
    
    correct_original = 0
    correct_adv = 0
    correct_purified = 0
    total = 0
    
    pixel_mse_list = []
    feature_distance_list = []
    
    with torch.no_grad():
        for adv_data, clean_data, labels in tqdm(test_loader, desc='Testing', leave=False):
            adv_data = adv_data.to(device)
            clean_data = clean_data.to(device)
            labels = labels.to(device)
            
            # 净化
            purified_data = denoiser(adv_data)
            
            # 计算像素MSE
            pixel_mse = F.mse_loss(purified_data, clean_data).item()
            pixel_mse_list.append(pixel_mse)
            
            # 提取特征并计算特征距离
            purified_seq = purified_data.squeeze()
            clean_seq = clean_data.squeeze()
            
            purified_feat = encoder(purified_seq)
            clean_feat = encoder(clean_seq)
            
            feature_distance = F.mse_loss(purified_feat, clean_feat).item()
            feature_distance_list.append(feature_distance)
            
            # VarCNN需要2通道输入
            if is_varcnn:
                # 转换clean数据
                clean_dir = torch.sign(clean_seq)
                clean_time = torch.abs(clean_seq)
                clean_time_diff = torch.diff(clean_time, dim=-1)
                clean_time_diff = torch.clamp(clean_time_diff, min=0)
                clean_time_diff = F.pad(clean_time_diff, (0, 1), value=0)
                clean_input = torch.stack([clean_dir, clean_time_diff], dim=1)
                
                # 转换adv数据
                adv_seq = adv_data.squeeze()
                adv_dir = torch.sign(adv_seq)
                adv_time = torch.abs(adv_seq)
                adv_time_diff = torch.diff(adv_time, dim=-1)
                adv_time_diff = torch.clamp(adv_time_diff, min=0)
                adv_time_diff = F.pad(adv_time_diff, (0, 1), value=0)
                adv_input = torch.stack([adv_dir, adv_time_diff], dim=1)
                
                # 转换purified数据
                purified_dir = torch.sign(purified_seq)
                purified_time = torch.abs(purified_seq)
                purified_time_diff = torch.diff(purified_time, dim=-1)
                purified_time_diff = torch.clamp(purified_time_diff, min=0)
                purified_time_diff = F.pad(purified_time_diff, (0, 1), value=0)
                purified_input = torch.stack([purified_dir, purified_time_diff], dim=1)
                
                # 分类
                clean_pred = classifier(clean_input).argmax(dim=1)
                adv_pred = classifier(adv_input).argmax(dim=1)
                purified_pred = classifier(purified_input).argmax(dim=1)
            else:
                # 分类
                clean_pred = classifier(clean_seq).argmax(dim=1)
                adv_seq = adv_data.squeeze()
                adv_pred = classifier(adv_seq).argmax(dim=1)
                purified_pred = classifier(purified_seq).argmax(dim=1)
            
            # 统计
            correct_original += (clean_pred == labels).sum().item()
            correct_adv += (adv_pred == labels).sum().item()
            correct_purified += (purified_pred == labels).sum().item()
            total += labels.size(0)
    
    # 计算指标
    results = {
        'pixel_mse': np.mean(pixel_mse_list),
        'feature_distance': np.mean(feature_distance_list),
        'acc_original': 100.0 * correct_original / total,
        'acc_adv': 100.0 * correct_adv / total,
        'acc_purified': 100.0 * correct_purified / total,
        'recovery_absolute': 100.0 * (correct_purified - correct_adv) / total,
        'recovery_relative': 100.0 * correct_purified / correct_original if correct_original > 0 else 0
    }
    
    return results


def test_single(defense, train_dataset, test_dataset, classifier_name, args, device):
    """测试单个组合"""
    print(f"\n{'='*80}")
    print(f"测试: {defense} - 训练集{train_dataset} - 测试集{test_dataset} - {classifier_name}")
    print(f"{'='*80}\n")
    
    # 1. 加载模型
    model_path = get_model_path(defense, train_dataset, args.model_dir, args.use_full)
    print(f"[OK] Model: {model_path}")
    
    denoiser = TransUNet(img_ch=1, output_ch=1).to(device)
    denoiser.load_state_dict(torch.load(model_path, map_location=device))
    denoiser.eval()
    
    # 2. 加载encoder
    encoder = IndependentEncoder(input_dim=512, output_dim=512).to(device)
    encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))
    encoder.eval()
    
    # 3. 确定类别数（根据测试数据集动态读取）
    if args.use_full:
        num_classes = 100
    else:
        # 从测试数据中读取实际的类别数
        test_data_path = os.path.join(args.data_path, 'splited_data', f'cw100_test_{test_dataset}.npz')
        test_data = np.load(test_data_path)
        max_label = test_data['labels'].max()
        num_classes = int(max_label) + 1
        print(f"[INFO] 数据集{test_dataset}标签范围: 0-{max_label}, num_classes={num_classes}")
    
    # 4. 加载分类器
    use_full = args.use_full
    classifier = load_classifier(classifier_name, num_classes, args.classifier_dir, test_dataset, device, use_full)
    print(f"[OK] Classifier: {classifier_name}_{test_dataset if not use_full else 'Full'}")
    
    # 5. 加载测试数据
    test_loader = get_adversarial_dataloader(
        datasets=None if args.use_full else [test_dataset],
        defense=defense,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        data_path=args.data_path,
        use_full=args.use_full
    )
    print(f"[OK] Data: {len(test_loader)} batches\n")
    
    # 6. 测试
    is_varcnn = (classifier_name == 'VarCNN')
    results = test_denoiser(denoiser, encoder, classifier, test_loader, device, is_varcnn)
    
    # 7. 打印结果
    print(f"\n{classifier_name} 结果:")
    print(f"  像素MSE: {results['pixel_mse']:.6f}")
    print(f"  特征距离: {results['feature_distance']:.6f}")
    print(f"  原始准确率: {results['acc_original']:.2f}%")
    print(f"  对抗准确率: {results['acc_adv']:.2f}%")
    print(f"  净化准确率: {results['acc_purified']:.2f}%")
    print(f"  绝对恢复: +{results['recovery_absolute']:.2f}%")
    print(f"  相对恢复: {results['recovery_relative']:.1f}%")
    
    return results


def save_cross_dataset_results(all_results, args, classifiers):
    """保存跨数据集测试结果"""
    result_dir = os.path.join(args.result_dir, 'splited')
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    result_file = os.path.join(
        result_dir,
        f'cross_dataset_{args.defense}_{args.train_dataset}_{timestamp}.json'
    )
    
    # 计算统计信息
    same_dataset_acc = []
    cross_dataset_acc = []
    
    for key, result_data in all_results.items():
        train_ds = result_data['train_dataset']
        test_ds = result_data['test_dataset']
        results = result_data['results']
        
        acc = results['acc_purified']
        if train_ds == test_ds:
            same_dataset_acc.append(acc)
        else:
            cross_dataset_acc.append(acc)
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'config': vars(args),
            'defense': args.defense,
            'train_dataset': args.train_dataset,
            'classifiers_tested': classifiers,
            'results': all_results,
            'summary': {
                'same_dataset_avg': np.mean(same_dataset_acc) if same_dataset_acc else None,
                'cross_dataset_avg': np.mean(cross_dataset_acc) if cross_dataset_acc else None,
                'generalization_gap': (np.mean(same_dataset_acc) - np.mean(cross_dataset_acc)) 
                                      if same_dataset_acc and cross_dataset_acc else None
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] 结果已保存: {result_file}")
    print(f"{'='*80}\n")


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 处理数据集参数
    if args.datasets and 'all' in args.datasets:
        args.datasets = ['A', 'B', 'C']
    elif not args.use_full and not args.datasets:
        args.datasets = ['A']  # 默认使用数据集A
    
    # 确定数据集信息
    if args.use_full:
        dataset_name = "完整数据集"
    else:
        dataset_name = "_".join(args.datasets)
    
    print(f"\n{'='*80}")
    print(f"测试改进的流量净化器")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"防御方法: {args.defense}")
    print(f"数据集: {dataset_name}")
    print(f"模型目录: {args.model_dir}")
    
    # 确定测试模式
    if args.cross_dataset:
        print(f"测试模式: 跨数据集测试")
    elif args.cross_classifier:
        print(f"测试模式: 跨分类器测试")
    else:
        print(f"测试模式: 单分类器测试 ({args.classifier})")
    print(f"{'='*80}\n")
    
    # 确定要测试的分类器
    if args.cross_classifier or args.cross_dataset:
        classifiers = ['VarCNN', 'DF', 'LSTMClassifier', 'TMWF']
    else:
        classifiers = [args.classifier]
    
    # 创建结果目录
    os.makedirs(args.result_dir, exist_ok=True)
    
    all_results = {}
    
    if args.test_all:
        # 测试所有组合
        defenses = ['mockingbird', 'WT', 'adv']
        datasets = args.datasets if not args.use_full else [None]
        
        for defense in defenses:
            for dataset in datasets:
                test_dataset = dataset if dataset else 'Full'
                for classifier in classifiers:
                    try:
                        results = test_single(defense, test_dataset, test_dataset, classifier, args, device)
                        key = f"{defense}_{test_dataset}_{classifier}"
                        all_results[key] = results
                    except Exception as e:
                        print(f"✗ 测试失败: {e}")
                        continue
    
    elif args.cross_dataset:
        # 跨数据集测试
        if not args.train_dataset:
            print("错误：跨数据集测试需要指定 --train_dataset")
            return
        
        # 检查是否测试所有组合
        if args.train_dataset.lower() == 'all':
            print("错误：Python脚本不支持 --train_dataset all")
            print("请使用批处理脚本或分别指定训练数据集：A、B 或 C")
            return
        
        # 验证训练数据集
        if args.train_dataset not in ['A', 'B', 'C']:
            print(f"错误：训练数据集必须是 A、B 或 C，不能是 '{args.train_dataset}'")
            return
        
        # 跨数据集测试：在所有数据集上测试
        all_test_datasets = ['A', 'B', 'C']
        
        print(f"训练数据集: {args.train_dataset}")
        print(f"测试数据集: {', '.join(all_test_datasets)}")
        print(f"分类器: {', '.join(classifiers)}\n")
        
        for test_ds in all_test_datasets:
            print(f"\n{'='*80}")
            print(f"测试数据集: {test_ds}")
            print(f"{'='*80}")
            
            for classifier in classifiers:
                try:
                    results = test_single(args.defense, args.train_dataset, test_ds, classifier, args, device)
                    key = f"train_{args.train_dataset}_test_{test_ds}_{classifier}"
                    all_results[key] = {
                        'train_dataset': args.train_dataset,
                        'test_dataset': test_ds,
                        'results': results
                    }
                except Exception as e:
                    print(f"[FAIL] 测试失败 ({classifier} on {test_ds}): {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # 保存跨数据集测试结果
        if all_results:
            save_cross_dataset_results(all_results, args, classifiers)
        else:
            print("\n[FAIL] 没有成功的测试结果")
        return
    
    elif args.cross_classifier:
        # 跨分类器测试
        test_dataset = args.datasets[0] if args.datasets else None
        
        for classifier in classifiers:
            try:
                results = test_single(args.defense, test_dataset, test_dataset, classifier, args, device)
                key = f"{args.defense}_{test_dataset}_{classifier}"
                all_results[key] = results
            except Exception as e:
                print(f"✗ 测试失败: {e}")
                continue
    
    else:
        # 单个测试
        test_dataset = args.datasets[0] if args.datasets else None
        results = test_single(args.defense, test_dataset, test_dataset, args.classifier, args, device)
        key = f"{args.defense}_{test_dataset}_{args.classifier}"
        all_results[key] = results
    
    # 保存结果
    if not args.cross_dataset:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 根据数据集类型确定保存路径
        if args.use_full:
            result_dir = args.result_dir
        elif args.datasets and len(args.datasets) == 1:
            result_dir = os.path.join(args.result_dir, 'splited')
        else:
            result_dir = args.result_dir
        
        os.makedirs(result_dir, exist_ok=True)
        
        if args.test_all:
            result_file = os.path.join(result_dir, f'test_all_{timestamp}.json')
        elif args.cross_classifier:
            result_file = os.path.join(result_dir, f'cross_classifier_{args.defense}_{dataset_name}_{timestamp}.json')
        else:
            result_file = os.path.join(result_dir, f'test_{args.defense}_{dataset_name}_{args.classifier}_{timestamp}.json')
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'config': vars(args),
                'results': all_results,
                'timestamp': timestamp
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*80}")
        print(f"✓ 结果已保存: {result_file}")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
