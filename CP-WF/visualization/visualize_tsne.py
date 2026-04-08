#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
t-SNE Visualization Analysis
Extract features from original traffic, defended traffic, and purified traffic, visualize using t-SNE dimensionality reduction
English version for paper publication
"""

import sys
import os

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'improved_denoiser'))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import argparse
from pathlib import Path

# Import models
from model.DF import DF
from refined_denoiser.src.encoder import IndependentEncoder
from refined_denoiser.src.denoiser import TransUNet
from refined_denoiser.src.data_loader import get_adversarial_dataloader
from calculate_tsne_metrics import (
    analyze_tsne_features, 
    print_metrics_table, 
    save_metrics
)

# Set English fonts and paper-level font sizes
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['figure.titlesize'] = 20



class FeatureExtractor:
    """Feature extractor, extracts features from the second-to-last layer of DF classifier"""
    
    def __init__(self, classifier):
        self.classifier = classifier
        self.features = None
        
        def hook_fn(module, input, output):
            self.features = output.detach()
        
        # classifier[5] is the second Linear layer (512->512)
        self.classifier.classifier[5].register_forward_hook(hook_fn)
    
    def extract(self, x):
        """Extract feature vectors"""
        with torch.no_grad():
            _ = self.classifier(x)
            return self.features.cpu().numpy()


def load_models(args):
    """Load all required models"""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Get project root directory
    project_root = Path(__file__).parent.parent
    
    # 1. Load classifier
    print(f"\nLoading classifier: {args.classifier}_{args.dataset}")
    if args.classifier == 'DF':
        classifier = DF(num_classes=args.num_classes)
    else:
        raise ValueError(f"Unsupported classifier: {args.classifier}")
    
    classifier_path = project_root / args.classifier_dir / 'split' / f'{args.classifier}_{args.dataset}.pth'
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier = classifier.to(device)
    classifier.eval()
    print(f"[OK] Classifier loaded")
    
    # 2. Load pretrained encoder
    print(f"\nLoading pretrained Encoder")
    encoder = IndependentEncoder(input_dim=512, output_dim=512)
    encoder_path = project_root / args.encoder_path
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder = encoder.to(device)
    encoder.eval()
    print(f"[OK] Encoder loaded")
    
    # 3. Load denoiser
    print(f"\nLoading denoiser: {args.defense}_dataset{args.dataset}")
    denoiser = TransUNet(img_ch=1, output_ch=1, num_heads=4, transformer_layers=1, dropout=0.1)
    denoiser_path = project_root / args.denoiser_dir / 'splited' / f'denoiser_{args.defense}_dataset{args.dataset}.pth'
    denoiser.load_state_dict(torch.load(denoiser_path, map_location=device))
    denoiser = denoiser.to(device)
    denoiser.eval()
    print(f"[OK] Denoiser loaded")
    
    return classifier, encoder, denoiser, device



def extract_features_for_classes(classifier, encoder, denoiser, test_loader, 
                                 selected_classes, samples_per_class, device):
    """Extract features for selected classes"""
    feature_extractor = FeatureExtractor(classifier)
    
    features_dict = {
        'original': {c: [] for c in selected_classes},
        'defended': {c: [] for c in selected_classes},
        'purified': {c: [] for c in selected_classes}
    }
    
    class_counts = {c: 0 for c in selected_classes}
    
    print(f"\nExtracting features...")
    print(f"Target classes: {selected_classes}")
    print(f"Samples per class: {samples_per_class}")
    
    with torch.no_grad():
        for batch_idx, (adv_batch, clean_batch, labels) in enumerate(tqdm(test_loader, desc="Extracting features")):
            # Fix dimensions: from [batch, 1, 1, 512] to [batch, 1, 512]
            adv_batch = adv_batch.squeeze(2).to(device)
            clean_batch = clean_batch.squeeze(2).to(device)
            labels = labels.to(device)
            
            if all(class_counts[c] >= samples_per_class for c in selected_classes):
                break
            
            for i in range(len(labels)):
                label = labels[i].item()
                
                if label not in selected_classes:
                    continue
                
                if class_counts[label] >= samples_per_class:
                    continue
                
                # Extract single sample
                adv_sample = adv_batch[i:i+1]
                clean_sample = clean_batch[i:i+1]
                
                # 1. Original traffic features
                feat_original = feature_extractor.extract(clean_sample)
                features_dict['original'][label].append(feat_original[0])
                
                # 2. Defended traffic features
                feat_defended = feature_extractor.extract(adv_sample)
                features_dict['defended'][label].append(feat_defended[0])
                
                # 3. Purified traffic features
                adv_sample_4d = adv_sample.unsqueeze(2)
                z = encoder(adv_sample.squeeze(1))
                purified = denoiser(adv_sample_4d)
                purified = purified.squeeze(2)
                feat_purified = feature_extractor.extract(purified)
                features_dict['purified'][label].append(feat_purified[0])
                
                class_counts[label] += 1
    
    # Convert to numpy arrays
    for state in ['original', 'defended', 'purified']:
        for c in selected_classes:
            if features_dict[state][c]:
                features_dict[state][c] = np.array(features_dict[state][c])
            else:
                print(f"Warning: No samples collected for class {c} in {state} state")
    
    print(f"\n[OK] Feature extraction completed")
    for c in selected_classes:
        print(f"  Class {c}: {len(features_dict['original'][c])} samples")
    
    return features_dict


def prepare_tsne_data(features_dict, selected_classes):
    """Prepare t-SNE input data"""
    X_list = []
    labels_list = []
    states_list = []
    
    state_map = {'original': 0, 'defended': 1, 'purified': 2}
    
    for state in ['original', 'defended', 'purified']:
        for class_id in selected_classes:
            features = features_dict[state][class_id]
            if len(features) > 0:
                X_list.append(features)
                labels_list.extend([class_id] * len(features))
                states_list.extend([state_map[state]] * len(features))
    
    X = np.vstack(X_list)
    labels = np.array(labels_list)
    states = np.array(states_list)
    
    print(f"\nt-SNE input data:")
    print(f"  Total samples: {len(X)}")
    print(f"  Feature dimensions: {X.shape[1]}")
    print(f"  Number of classes: {len(selected_classes)}")
    print(f"  Number of states: 3 (original, defended, purified)")
    
    return X, labels, states


def visualize_tsne(X, labels, states, selected_classes, args):
    """Perform t-SNE dimensionality reduction and visualization"""
    print(f"\nPerforming t-SNE dimensionality reduction...")
    print(f"  Parameters: perplexity={args.perplexity}, max_iter={args.n_iter}")
    
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        max_iter=args.n_iter,
        random_state=42,
        verbose=1
    )
    
    X_embedded = tsne.fit_transform(X)
    
    print(f"[OK] t-SNE dimensionality reduction completed")
    
    # Create visualizations
    create_visualizations(X_embedded, labels, states, selected_classes, args)



def create_visualizations(X_embedded, labels, states, selected_classes, args):
    """Create multiple visualization charts"""
    output_dir = Path(__file__).parent / 'results' / 'tsne_en'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Color mapping
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_classes)))
    class_colors = {c: colors[i] for i, c in enumerate(selected_classes)}
    
    # State markers
    state_markers = {0: 'o', 1: 'x', 2: '^'}
    state_names = {0: 'Original', 1: 'Defended', 2: 'Purified'}
    
    # 1. Visualization grouped by state
    print(f"\nGenerating visualization grouped by state...")
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    
    for state_id, ax in enumerate(axes):
        mask = states == state_id
        X_state = X_embedded[mask]
        labels_state = labels[mask]
        
        for class_id in selected_classes:
            class_mask = labels_state == class_id
            ax.scatter(
                X_state[class_mask, 0],
                X_state[class_mask, 1],
                c=[class_colors[class_id]],
                label=f'Class {class_id}',
                alpha=0.6,
                s=80,
                edgecolors='black',
                linewidths=1.0
            )
        
        ax.set_title(f'{state_names[state_id]} Traffic', fontsize=20, pad=15, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1', fontsize=18)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=18)
        ax.legend(fontsize=14, loc='best', ncol=2, framealpha=0.9)
        ax.grid(alpha=0.3, linestyle='--', linewidth=1.0)
        ax.tick_params(labelsize=16)
    
    plt.suptitle(f't-SNE Feature Space Visualization', 
                 fontsize=22, y=1.02, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_dir / f'tsne_by_state_{args.defense}_{args.dataset}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_file.name}")
    plt.close()
    
    # 2. Visualization with all states overlaid
    print(f"\nGenerating visualization with all states overlaid...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for class_id in selected_classes:
        for state_id in [0, 1, 2]:
            mask = (labels == class_id) & (states == state_id)
            ax.scatter(
                X_embedded[mask, 0],
                X_embedded[mask, 1],
                c=[class_colors[class_id]],
                marker=state_markers[state_id],
                alpha=0.5,
                s=50 if state_id == 2 else 30,
                edgecolors='black',
                linewidths=0.5
            )
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = []
    
    for class_id in selected_classes:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor=class_colors[class_id], 
                   markersize=8, label=f'Class {class_id}')
        )
    
    legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor='gray', markersize=8, 
                                  label='Original Traffic'))
    legend_elements.append(Line2D([0], [0], marker='x', color='gray', 
                                  markersize=8, label='Defended Traffic'))
    legend_elements.append(Line2D([0], [0], marker='^', color='w', 
                                  markerfacecolor='gray', markersize=8, 
                                  label='Purified Traffic'))
    
    ax.legend(handles=legend_elements, fontsize=10, loc='best', ncol=2)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title(f't-SNE Feature Space Visualization - All States ({args.defense} + {args.classifier})', 
                 fontsize=14, pad=10)
    ax.grid(alpha=0.3, linestyle='--')
    
    output_file = output_dir / f'tsne_all_states_{args.defense}_{args.dataset}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_file.name}")
    plt.close()

    
    # 3. Visualization grouped by class
    print(f"\nGenerating visualization grouped by class...")
    n_classes = len(selected_classes)
    n_cols = min(5, n_classes)
    n_rows = (n_classes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_classes == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes
    
    for idx, class_id in enumerate(selected_classes):
        ax = axes[idx]
        
        for state_id in [0, 1, 2]:
            mask = (labels == class_id) & (states == state_id)
            ax.scatter(
                X_embedded[mask, 0],
                X_embedded[mask, 1],
                marker=state_markers[state_id],
                label=state_names[state_id],
                alpha=0.6,
                s=50,
                edgecolors='black',
                linewidths=0.5
            )
        
        ax.set_title(f'Class {class_id}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, linestyle='--')
    
    # Hide extra subplots
    for idx in range(n_classes, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f't-SNE Feature Space Visualization - By Class ({args.defense} + {args.classifier})', 
                 fontsize=16, y=1.00)
    plt.tight_layout()
    
    output_file = output_dir / f'tsne_by_class_{args.defense}_{args.dataset}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_file.name}")
    plt.close()
    
    print(f"\n[OK] All visualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='t-SNE Feature Visualization')
    
    # Data and model parameters
    parser.add_argument('--dataset', type=str, default='A', choices=['A', 'B', 'C'])
    parser.add_argument('--defense', type=str, default='WT', choices=['adv', 'WT', 'mockingbird'])
    parser.add_argument('--classifier', type=str, default='DF')
    parser.add_argument('--num_classes', type=int, default=100)
    
    # Path parameters
    parser.add_argument('--data_path', type=str, default='processed_data')
    parser.add_argument('--classifier_dir', type=str, default='saved_models')
    parser.add_argument('--encoder_path', type=str,
                        default='improved_denoiser/saved_models/pretrained_encoder.pth')
    parser.add_argument('--denoiser_dir', type=str,
                        default='improved_denoiser/saved_models')
    
    # Sampling parameters
    parser.add_argument('--selected_classes', type=int, nargs='+',
                        default=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
    parser.add_argument('--samples_per_class', type=int, default=100)
    
    # t-SNE parameters
    parser.add_argument('--perplexity', type=int, default=30)
    parser.add_argument('--n_iter', type=int, default=1000)
    
    # Other parameters
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda:0')
    
    args = parser.parse_args()
    
    print("="*80)
    print("t-SNE Feature Space Visualization")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Defense Method: {args.defense}")
    print(f"Classifier: {args.classifier}")
    print(f"Selected Classes: {args.selected_classes}")
    print(f"Samples per Class: {args.samples_per_class}")
    print("="*80)
    
    # Load models
    classifier, encoder, denoiser, device = load_models(args)
    
    # Load data
    print(f"\nLoading test data...")
    data_path = Path(__file__).parent.parent / args.data_path
    test_loader = get_adversarial_dataloader(
        datasets=[args.dataset],
        defense=args.defense,
        batch_size=args.batch_size,
        shuffle=False,
        data_path=str(data_path),
        use_full=False
    )
    print(f"[OK] Data loaded")
    
    # Extract features
    features_dict = extract_features_for_classes(
        classifier, encoder, denoiser, test_loader,
        args.selected_classes, args.samples_per_class, device
    )
    
    # Prepare t-SNE data
    X, labels, states = prepare_tsne_data(features_dict, args.selected_classes)
    
    # Perform t-SNE and visualize
    visualize_tsne(X, labels, states, args.selected_classes, args)
    
    # Calculate quantitative metrics
    print("\n" + "="*80)
    print("Calculating quantitative metrics...")
    print("="*80)
    
    metrics = analyze_tsne_features(features_dict, args.selected_classes)
    print_metrics_table(metrics, f"{args.defense} (Dataset {args.dataset})")
    
    # Save metrics
    output_dir = Path(__file__).parent / 'results' / 'tsne_en'
    metrics_file = output_dir / f'metrics_{args.defense}_{args.dataset}.json'
    save_metrics(metrics, metrics_file)
    
    print("\n" + "="*80)
    print("t-SNE visualization completed!")
    print("="*80)


if __name__ == '__main__':
    main()
