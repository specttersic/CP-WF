#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCA Visualization Analysis
Extract features from original traffic, defended traffic, and purified traffic, visualize using PCA dimensionality reduction
PCA preserves global geometric structure, verifying inter-class separability
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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
import argparse
from pathlib import Path
import json

# Import models
from model.DF import DF
from refined_denoiser.src.encoder import IndependentEncoder
from refined_denoiser.src.denoiser import TransUNet
from refined_denoiser.src.data_loader import get_adversarial_dataloader

# Set English fonts and paper-level font sizes
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14  # Base font size
plt.rcParams['axes.labelsize'] = 16  # Axis labels
plt.rcParams['axes.titlesize'] = 18  # Subplot titles
plt.rcParams['xtick.labelsize'] = 14  # x-axis ticks
plt.rcParams['ytick.labelsize'] = 14  # y-axis ticks
plt.rcParams['legend.fontsize'] = 13  # Legend
plt.rcParams['figure.titlesize'] = 20  # Main title
plt.rcParams['lines.linewidth'] = 2  # Line width
plt.rcParams['lines.markersize'] = 8  # Marker size


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
    classifier = DF(num_classes=args.num_classes)
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
    
    print(f"\n[OK] Feature extraction completed")
    for c in selected_classes:
        print(f"  Class {c}: {len(features_dict['original'][c])} samples")
    
    return features_dict


def prepare_pca_data(features_dict, selected_classes):
    """Prepare PCA input data"""
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
    
    print(f"\nPCA input data:")
    print(f"  Total samples: {len(X)}")
    print(f"  Feature dimensions: {X.shape[1]}")
    print(f"  Number of classes: {len(selected_classes)}")
    print(f"  Number of states: 3 (original, defended, purified)")
    
    return X, labels, states


def visualize_pca(X, labels, states, selected_classes, args):
    """Perform PCA dimensionality reduction and visualization"""
    print(f"\nPerforming PCA dimensionality reduction...")
    print(f"  Reducing: {X.shape[1]}D → 2D")
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    # Print explained variance ratio
    explained_var = pca.explained_variance_ratio_
    print(f"[OK] PCA dimensionality reduction completed")
    print(f"  PC1 explained variance ratio: {explained_var[0]:.2%}")
    print(f"  PC2 explained variance ratio: {explained_var[1]:.2%}")
    print(f"  Cumulative explained variance: {explained_var.sum():.2%}")
    
    # Save PCA information
    pca_info = {
        'explained_variance_ratio': [float(v) for v in explained_var.tolist()],
        'cumulative_variance': float(explained_var.sum()),
        'n_components': 2,
        'n_features': int(X.shape[1]),
        'n_samples': int(X.shape[0])
    }
    
    # Create visualizations
    create_visualizations(X_pca, labels, states, selected_classes, args, pca_info)


def create_visualizations(X_pca, labels, states, selected_classes, args, pca_info):
    """Create multiple visualization charts"""
    output_dir = Path(__file__).parent / 'results' / 'pca_en'
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
        X_state = X_pca[mask]
        labels_state = labels[mask]
        
        for class_id in selected_classes:
            class_mask = labels_state == class_id
            ax.scatter(
                X_state[class_mask, 0],
                X_state[class_mask, 1],
                c=[class_colors[class_id]],
                label=f'Class {class_id}',
                alpha=0.6,
                s=80,  # Larger markers
                edgecolors='black',
                linewidths=1.0  # Thicker edges
            )
        
        ax.set_title(f'{state_names[state_id]} Traffic', fontsize=20, pad=15, fontweight='bold')
        ax.set_xlabel(f'PC1 ({pca_info["explained_variance_ratio"][0]:.1%})', fontsize=18)
        ax.set_ylabel(f'PC2 ({pca_info["explained_variance_ratio"][1]:.1%})', fontsize=18)
        ax.legend(fontsize=14, loc='best', ncol=2, framealpha=0.9)
        ax.grid(alpha=0.3, linestyle='--', linewidth=1.0)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=1.0, alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=1.0, alpha=0.3)
        ax.tick_params(labelsize=16)
    
    plt.suptitle(f'PCA Feature Space Visualization', 
                 fontsize=22, y=1.02, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_dir / f'pca_by_state_{args.defense}_{args.dataset}.png'
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
                X_pca[mask, 0],
                X_pca[mask, 1],
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
    ax.set_xlabel(f'PC1 ({pca_info["explained_variance_ratio"][0]:.1%})', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca_info["explained_variance_ratio"][1]:.1%})', fontsize=12)
    ax.set_title(f'PCA Feature Space Visualization - All States', 
                 fontsize=14, pad=10)
    ax.grid(alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    output_file = output_dir / f'pca_all_states_{args.defense}_{args.dataset}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_file.name}")
    plt.close()

    
    # 3. Explained variance ratio chart
    print(f"\nGenerating explained variance ratio chart...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    components = ['PC1', 'PC2']
    variance_ratios = pca_info['explained_variance_ratio']
    
    bars = ax.bar(components, variance_ratios, color=['#4472C4', '#ED7D31'], alpha=0.8)
    ax.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax.set_title(f'PCA Principal Component Explained Variance Ratio ({args.defense})', fontsize=14, pad=10)
    ax.set_ylim([0, max(variance_ratios) * 1.2])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value annotations
    for bar, ratio in zip(bars, variance_ratios):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{ratio:.1%}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add cumulative variance
    ax.text(0.5, max(variance_ratios) * 1.1, 
            f'Cumulative: {pca_info["cumulative_variance"]:.1%}',
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    output_file = output_dir / f'pca_variance_{args.defense}_{args.dataset}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_file.name}")
    plt.close()
    
    # Save PCA information
    info_file = output_dir / f'pca_info_{args.defense}_{args.dataset}.json'
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(pca_info, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved PCA info: {info_file.name}")
    
    print(f"\n[OK] All visualizations saved to: {output_dir}")



def main():
    parser = argparse.ArgumentParser(description='PCA Feature Visualization')
    
    # Data and model parameters
    parser.add_argument('--dataset', type=str, default='A', choices=['A', 'B', 'C'])
    parser.add_argument('--defense', type=str, default='adv', choices=['adv', 'WT', 'mockingbird'])
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
    
    # Other parameters
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda:0')
    
    args = parser.parse_args()
    
    print("="*80)
    print("PCA Feature Space Visualization")
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
    # Adjust data_path to be relative to project root
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
    
    # Prepare PCA data
    X, labels, states = prepare_pca_data(features_dict, args.selected_classes)
    
    # Perform PCA and visualize
    visualize_pca(X, labels, states, args.selected_classes, args)
    
    print("\n" + "="*80)
    print("PCA visualization completed!")
    print("="*80)


if __name__ == '__main__':
    main()
