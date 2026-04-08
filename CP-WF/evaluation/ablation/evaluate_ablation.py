"""
消融研究评估脚本

评估所有消融变体的净化准确率，输出对比表格。

用法：
  # 评估单个变体
  python evaluate_ablation.py --variant wo_per --defense adv

  # 评估所有变体（生成完整对比表格）
  python evaluate_ablation.py --all --defense adv
"""

import sys, os

script_dir   = os.path.dirname(os.path.abspath(__file__))
improved_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(improved_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, improved_dir)

import torch
import numpy as np
import argparse
from tqdm import tqdm
import json
from datetime import datetime

from model.DF import DF
from refined_denoiser.src.encoder import IndependentEncoder
from refined_denoiser.src.denoiser import TransUNet
from refined_denoiser.src.data_loader import get_adversarial_dataloader
from improved_denoiser.ablation.train_ablation import TransUNetNoAG


# ──────────────────────────────────────────────
VARIANTS = [
    ('full',     False, '完整模型 (L_rec+L_per+L_con)'),
    ('wo_per',   False, 'w/o L_per (L_rec+L_con)'),
    ('wo_con',   False, 'w/o L_con (L_rec+L_per)'),
    ('only_rec', False, '仅 L_rec'),
    ('full',     True,  'w/o AG (L_rec+L_per+L_con)'),
]


def parse_args():
    parser = argparse.ArgumentParser(description='消融研究评估脚本')

    parser.add_argument('--all', action='store_true',
                        help='评估所有消融变体')
    parser.add_argument('--variant', type=str, default=None,
                        choices=['full', 'wo_per', 'wo_con', 'only_rec'],
                        help='单独评估某个变体')
    parser.add_argument('--no_ag', action='store_true',
                        help='评估 w/o AG 变体')
    parser.add_argument('--defense', type=str, default='adv',
                        choices=['adv', 'WT', 'mockingbird'])
    parser.add_argument('--classifier', type=str, default='DF')
    parser.add_argument('--use_full', action='store_true', default=True)
    parser.add_argument('--data_path', type=str, default='../../processed_data')
    parser.add_argument('--encoder_path', type=str,
                        default='../saved_models/pretrained_encoder.pth')
    parser.add_argument('--model_dir', type=str,
                        default='../saved_models/ablation')
    parser.add_argument('--classifier_dir', type=str, default='../../saved_models')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--result_dir', type=str, default='../results/ablation')
    parser.add_argument('--device', type=str, default='cuda:0')

    return parser.parse_args()


def load_denoiser(model_dir, defense, ablation, no_ag, device):
    """加载指定变体的 denoiser"""
    variant = ablation
    if no_ag:
        variant = f'{ablation}_no_ag'

    model_path = os.path.join(model_dir, f'denoiser_{defense}_{variant}.pth')

    if not os.path.exists(model_path):
        return None, model_path

    if no_ag:
        denoiser = TransUNetNoAG(img_ch=1, output_ch=1).to(device)
    else:
        denoiser = TransUNet(img_ch=1, output_ch=1).to(device)

    denoiser.load_state_dict(torch.load(model_path, map_location=device))
    denoiser.eval()
    return denoiser, model_path


def evaluate_variant(denoiser, classifier, test_loader, device):
    """评估单个变体，返回 (acc_clean, acc_adv, acc_purified, mse)"""
    correct_clean = correct_adv = correct_purified = 0
    total_mse = total = 0

    with torch.no_grad():
        for adv, clean, labels in tqdm(test_loader, desc='Evaluating', leave=False):
            adv, clean, labels = adv.to(device), clean.to(device), labels.to(device)
            bs = adv.shape[0]

            purified = denoiser(adv)

            mse = torch.mean((purified - clean) ** 2).item()
            total_mse += mse * bs

            clean_seq    = clean.squeeze()
            adv_seq      = adv.squeeze()
            purified_seq = purified.squeeze()

            correct_clean    += (classifier(clean_seq).argmax(1) == labels).sum().item()
            correct_adv      += (classifier(adv_seq).argmax(1)   == labels).sum().item()
            correct_purified += (classifier(purified_seq).argmax(1) == labels).sum().item()
            total += bs

    return (100.0 * correct_clean    / total,
            100.0 * correct_adv      / total,
            100.0 * correct_purified / total,
            total_mse / total)


def print_table(rows, baseline_acc):
    """打印消融结果表格"""
    header = f"{'变体':<35} {'净化准确率':>10} {'vs 完整模型':>12} {'MSE':>10}"
    print("\n" + "=" * 72)
    print("消融研究结果")
    print("=" * 72)
    print(header)
    print("-" * 72)
    for name, acc_purified, mse in rows:
        delta = acc_purified - baseline_acc
        marker = " ◀ 基准" if delta == 0 else ""
        print(f"{name:<35} {acc_purified:>9.2f}%  {delta:>+10.2f}%  {mse:>10.6f}{marker}")
    print("=" * 72)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ── 加载分类器 ──
    num_classes = 100
    clf_path = os.path.join(args.classifier_dir, f'{args.classifier}.pth')
    classifier = DF(num_classes=num_classes)
    classifier.load_state_dict(torch.load(clf_path, map_location=device))
    classifier.to(device).eval()
    print(f"✓ Classifier: {clf_path}")

    # ── 加载测试数据 ──
    test_loader = get_adversarial_dataloader(
        datasets=None,
        defense=args.defense,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        data_path=args.data_path,
        use_full=args.use_full,
    )
    print(f"✓ Test data: {len(test_loader)} batches")

    # ── 确定要评估的变体列表 ──
    if args.all:
        to_eval = VARIANTS
    else:
        ablation = args.variant or 'full'
        label = next((v[2] for v in VARIANTS
                      if v[0] == ablation and v[1] == args.no_ag), ablation)
        to_eval = [(ablation, args.no_ag, label)]

    # ── 逐个评估 ──
    rows = []
    baseline_acc = None
    all_results = {}

    for ablation, no_ag, label in to_eval:
        denoiser, model_path = load_denoiser(
            args.model_dir, args.defense, ablation, no_ag, device)

        if denoiser is None:
            print(f"\n⚠ 模型未找到: {model_path}，跳过 [{label}]")
            continue

        print(f"\n评估: {label}")
        acc_clean, acc_adv, acc_purified, mse = evaluate_variant(
            denoiser, classifier, test_loader, device)

        print(f"  干净准确率:  {acc_clean:.2f}%")
        print(f"  对抗准确率:  {acc_adv:.2f}%")
        print(f"  净化准确率:  {acc_purified:.2f}%")
        print(f"  MSE:         {mse:.6f}")

        rows.append((label, acc_purified, mse))
        all_results[label] = {
            'acc_clean': acc_clean,
            'acc_adv': acc_adv,
            'acc_purified': acc_purified,
            'mse': mse,
        }

        # 完整模型作为基准
        if ablation == 'full' and not no_ag:
            baseline_acc = acc_purified

    # ── 打印汇总表格 ──
    if rows:
        print_table(rows, baseline_acc or rows[0][1])

    # ── 保存结果 ──
    os.makedirs(args.result_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(
        args.result_dir,
        f'ablation_{args.defense}_{args.classifier}_{ts}.json')

    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump({
            'defense': args.defense,
            'classifier': args.classifier,
            'timestamp': ts,
            'results': all_results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 结果已保存: {result_path}\n")


if __name__ == '__main__':
    main()
