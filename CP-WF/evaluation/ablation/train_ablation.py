"""
消融研究训练脚本

支持四种消融变体：
  full      - 完整三重损失 (L_rec + L_per + L_con)
  wo_per    - 移除感知损失 (L_rec + L_con)
  wo_con    - 移除对比损失 (L_rec + L_per)
  only_rec  - 仅重建损失  (L_rec)

w/o AG 变体通过 --no_ag 标志控制（使用标准 U-Net 跳跃连接）

用法示例：
  python train_ablation.py --ablation full     --defense adv
  python train_ablation.py --ablation wo_per   --defense adv
  python train_ablation.py --ablation wo_con   --defense adv
  python train_ablation.py --ablation only_rec --defense adv
  python train_ablation.py --ablation full     --defense adv --no_ag
"""

import sys, os

script_dir   = os.path.dirname(os.path.abspath(__file__))
improved_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(improved_dir)
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

from refined_denoiser.src.encoder import IndependentEncoder
from refined_denoiser.src.denoiser import TransUNet
from refined_denoiser.src.data_loader import get_adversarial_dataloader
from improved_denoiser.ablation.ablation_losses import AblationLoss


# ──────────────────────────────────────────────
# 不含 AG 的标准 U-Net（用于 w/o AG 消融）
# ──────────────────────────────────────────────
class TransUNetNoAG(TransUNet):
    """移除 Attention Gate，改用标准直接拼接跳跃连接"""

    def forward(self, x):
        # 编码路径
        x1 = self.Conv1(x)
        x2 = self.Conv2(self.Maxpool(x1))
        x3 = self.Conv3(self.Maxpool(x2))
        x4 = self.Conv4(self.Maxpool(x3))
        x5 = self.Conv5(self.Maxpool(x4))

        # Transformer 瓶颈
        x5 = self.transformer(x5)

        # 解码路径（直接拼接，不经过 AG）
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        return self.Conv_1x1(d2)


# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description='消融研究训练脚本')

    parser.add_argument('--ablation', type=str, required=True,
                        choices=['full', 'wo_per', 'wo_con', 'only_rec'],
                        help='消融模式')
    parser.add_argument('--no_ag', action='store_true',
                        help='移除 Attention Gate（w/o AG 消融）')
    parser.add_argument('--defense', type=str, default='adv',
                        choices=['adv', 'WT', 'mockingbird'],
                        help='防御方法（默认 adv）')
    parser.add_argument('--use_full', action='store_true', default=False,
                        help='使用完整数据集（默认 False，使用 --datasets 指定子数据集）')
    parser.add_argument('--datasets', type=str, nargs='+', default=None)
    parser.add_argument('--data_path', type=str, default='../../processed_data')
    parser.add_argument('--encoder_path', type=str,
                        default='../saved_models/pretrained_encoder.pth')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta',  type=float, default=0.25)
    parser.add_argument('--gamma', type=float, default=0.25)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--save_dir', type=str, default='../saved_models/ablation')
    parser.add_argument('--result_dir', type=str, default='../results/ablation')
    parser.add_argument('--device', type=str, default='cuda:0')

    return parser.parse_args()


def train_one_epoch(denoiser, encoder, loader, criterion, optimizer, device):
    denoiser.train()
    encoder.eval()

    total_loss = 0
    details = {'mse': 0, 'perceptual': 0, 'supcon': 0}

    pbar = tqdm(loader, desc='Training', leave=False)
    for adv, clean, labels in pbar:
        adv, clean, labels = adv.to(device), clean.to(device), labels.to(device)

        purified = denoiser(adv)
        loss, loss_dict = criterion(purified, clean, labels, encoder)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        for k in details:
            details[k] += loss_dict.get(k, 0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}',
                          'mse':  f'{loss_dict["mse"]:.4f}'})

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in details.items()}


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 变体名称（用于文件命名）
    variant = args.ablation
    if args.no_ag:
        variant = f'{variant}_no_ag'

    print(f"\n{'='*70}")
    print(f"消融研究训练  |  变体: {variant}  |  防御: {args.defense}")
    print(f"{'='*70}\n")

    # ── 加载 Encoder ──
    encoder = IndependentEncoder(input_dim=512, output_dim=512).to(device)
    if os.path.exists(args.encoder_path):
        encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))
        print(f"✓ Encoder loaded: {args.encoder_path}")
    else:
        print(f"⚠ Encoder not found, using random init")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # ── 创建 Denoiser ──
    if args.no_ag:
        denoiser = TransUNetNoAG(img_ch=1, output_ch=1).to(device)
        print("✓ Denoiser: TransUNet (w/o AG)")
    else:
        denoiser = TransUNet(img_ch=1, output_ch=1).to(device)
        print("✓ Denoiser: TransUNet (with AG)")

    # ── 加载数据 ──
    train_loader = get_adversarial_dataloader(
        datasets=args.datasets,
        defense=args.defense,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        data_path=args.data_path,
        use_full=args.use_full,
    )
    print(f"✓ Data: {len(train_loader)} batches\n")

    # ── 损失函数 ──
    criterion = AblationLoss(
        mode=args.ablation,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        temperature=args.temperature,
    )

    # ── 优化器 ──
    optimizer = Adam(denoiser.parameters(), lr=args.lr,
                     weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── 训练循环 ──
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    best_loss = float('inf')
    model_path = os.path.join(args.save_dir,
                              f'denoiser_{args.defense}_{variant}.pth')
    history = []

    print(f"\n开始训练 ({args.epochs} epochs)...\n")
    for epoch in range(1, args.epochs + 1):
        avg_loss, details = train_one_epoch(
            denoiser, encoder, train_loader, criterion, optimizer, device)

        scheduler.step()

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"loss={avg_loss:.4f}  "
              f"mse={details['mse']:.4f}  "
              f"per={details['perceptual']:.4f}  "
              f"con={details['supcon']:.4f}")

        history.append({'epoch': epoch, 'total_loss': avg_loss, **details})

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(denoiser.state_dict(), model_path)
            print(f"  ✓ Best model saved → {model_path}")

    # ── 保存训练历史 ──
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    hist_path = os.path.join(args.result_dir,
                             f'history_{args.defense}_{variant}_{ts}.json')
    with open(hist_path, 'w', encoding='utf-8') as f:
        json.dump({'variant': variant, 'defense': args.defense,
                   'best_loss': best_loss, 'model_path': model_path,
                   'history': history}, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 训练完成  best_loss={best_loss:.4f}")
    print(f"✓ 模型: {model_path}")
    print(f"✓ 历史: {hist_path}\n")


if __name__ == '__main__':
    main()
