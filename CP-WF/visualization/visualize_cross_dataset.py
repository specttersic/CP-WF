"""
Visualize Cross-Dataset Experimental Results - Version 2
Display adversarial accuracy and purified accuracy for each classifier on different test sets
Use different bar styles to distinguish different datasets
English version for paper publication
"""
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# Set English fonts and paper-level font sizes
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['figure.titlesize'] = 20

def load_complete_results():
    """Load results from complete JSON files"""
    # Get script directory and construct correct path
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / 'improved_denoiser' / 'results' / 'splited'
    
    data = {}
    
    # Load complete results for three defense methods
    defense_files = {
        'Advtraffic': 'cross_dataset_adv_A_complete.json',
        'Walkie-Talkie': 'cross_dataset_WT_A_complete.json',
        'Mockingbird': 'cross_dataset_mockingbird_A_complete.json'
    }
    
    for defense_name, filename in defense_files.items():
        filepath = results_dir / filename
        with open(filepath, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # Reorganize data structure
        defense_data = {
            'test_A': {},
            'test_B': {},
            'test_C': {}
        }
        
        # Extract data from results
        for key, value in json_data['results'].items():
            test_ds = value['test_dataset']
            results = value['results']
            
            # Extract classifier name
            if 'VarCNN' in key:
                clf = 'VarCNN'
            elif 'DF' in key:
                clf = 'DF'
            elif 'LSTM' in key:
                clf = 'LSTM'
            elif 'TMWF' in key:
                clf = 'TMWF'
            else:
                continue
            
            test_key = f'test_{test_ds}'
            defense_data[test_key][clf] = {
                'adv': results['acc_adv'],
                'purified': results['acc_purified']
            }
        
        data[defense_name] = defense_data
    
    return data


def plot_cross_dataset_performance(defense_name, data, save_path=None):
    """
    Plot cross-dataset performance for a single defense method
    Display adversarial accuracy and purified accuracy for each classifier on different test sets
    """
    fig, ax = plt.subplots(figsize=(15, 7))
    
    test_datasets = ['A', 'B', 'C']
    classifiers = ['VarCNN', 'DF', 'LSTM', 'TMWF']
    
    # Create a group of bars for each classifier
    x = np.arange(len(classifiers))
    width = 0.13  # 6 bars (3 test sets × 2 accuracy types)
    
    # Color configuration - use more distinct colors
    test_colors = {
        'A': '#3498db',  # Blue
        'B': '#e74c3c',  # Red
        'C': '#2ecc71'   # Green
    }
    
    # Bar styles: adversarial accuracy with diagonal lines, purified accuracy solid
    hatches = {
        'adv': '///',   # Adversarial accuracy: diagonal lines
        'purified': ''  # Purified accuracy: solid
    }
    
    # Draw bars for each test dataset
    for test_idx, test_ds in enumerate(test_datasets):
        test_key = f'test_{test_ds}'
        
        # Extract adversarial and purified accuracies
        adv_accs = []
        purified_accs = []
        
        for clf in classifiers:
            if clf in data[test_key]:
                adv_accs.append(data[test_key][clf]['adv'])
                purified_accs.append(data[test_key][clf]['purified'])
            else:
                adv_accs.append(0)
                purified_accs.append(0)
        
        # Calculate bar positions
        offset_adv = width * (test_idx * 2 - 2)
        offset_purified = width * (test_idx * 2 - 1)
        
        # Draw adversarial accuracy (diagonal line fill)
        bars_adv = ax.bar(x + offset_adv, adv_accs, width,
                         label=f'Dataset {test_ds} - Defended',
                         color=test_colors[test_ds], alpha=0.5,
                         edgecolor='black', linewidth=0.8,
                         hatch=hatches['adv'])
        
        # Draw purified accuracy (solid fill)
        bars_purified = ax.bar(x + offset_purified, purified_accs, width,
                              label=f'Dataset {test_ds} - Purified',
                              color=test_colors[test_ds], alpha=0.9,
                              edgecolor='black', linewidth=0.8,
                              hatch=hatches['purified'])
        
        # Add value labels (only on purified accuracy)
        for bar, acc in zip(bars_purified, purified_accs):
            if acc > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{acc:.0f}',
                       ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    # Set labels and title
    ax.set_xlabel('Classifier', fontsize=13, fontweight='bold', labelpad=15)
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'{defense_name} Defense - Cross-Dataset Performance Comparison', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(classifiers, fontsize=12, fontweight='bold')
    
    # Simplified legend: only show color and style meanings
    from matplotlib.patches import Patch
    
    # Create custom legend elements
    legend_elements = [
        # Dataset colors
        Patch(facecolor=test_colors['A'], edgecolor='black', label='Dataset A'),
        Patch(facecolor=test_colors['B'], edgecolor='black', label='Dataset B'),
        Patch(facecolor=test_colors['C'], edgecolor='black', label='Dataset C'),
        # Style descriptions
        Patch(facecolor='gray', edgecolor='black', hatch='///', alpha=0.5, label='Defended Accuracy'),
        Patch(facecolor='gray', edgecolor='black', alpha=0.9, label='Purified Accuracy'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12), 
             ncol=5, fontsize=10, framealpha=0.9, edgecolor='black')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 110)
    
    # Add horizontal reference line
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(len(classifiers)-0.2, 91, '90%', fontsize=9, color='gray')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Chart saved to: {save_path}")
    
    plt.close()


def plot_all_defenses_comparison(all_data, save_path=None):
    """Plot comprehensive comparison of all defense methods"""
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    defense_names = ['Advtraffic', 'Walkie-Talkie', 'Mockingbird']
    test_datasets = ['A', 'B', 'C']
    classifiers = ['VarCNN', 'DF', 'LSTM', 'TMWF']
    
    # Color configuration
    test_colors = {
        'A': '#3498db',
        'B': '#e74c3c',
        'C': '#2ecc71'
    }
    
    hatches = {
        'adv': '///',
        'purified': ''
    }
    
    for def_idx, (defense_name, ax) in enumerate(zip(defense_names, axes)):
        data = all_data[defense_name]
        
        x = np.arange(len(classifiers))
        width = 0.13
        
        # Draw bars for each test dataset
        for test_idx, test_ds in enumerate(test_datasets):
            test_key = f'test_{test_ds}'
            
            adv_accs = []
            purified_accs = []
            
            for clf in classifiers:
                if clf in data[test_key]:
                    adv_accs.append(data[test_key][clf]['adv'])
                    purified_accs.append(data[test_key][clf]['purified'])
                else:
                    adv_accs.append(0)
                    purified_accs.append(0)
            
            offset_adv = width * (test_idx * 2 - 2)
            offset_purified = width * (test_idx * 2 - 1)
            
            # Only show legend in first subplot
            show_label = (def_idx == 1)
            
            ax.bar(x + offset_adv, adv_accs, width,
                  label=f'Dataset {test_ds} - Defended' if show_label else '',
                  color=test_colors[test_ds], alpha=0.5,
                  edgecolor='black', linewidth=0.7,
                  hatch=hatches['adv'])
            
            bars_purified = ax.bar(x + offset_purified, purified_accs, width,
                                  label=f'Dataset {test_ds} - Purified' if show_label else '',
                                  color=test_colors[test_ds], alpha=0.9,
                                  edgecolor='black', linewidth=0.7,
                                  hatch=hatches['purified'])
            
            # Add value labels
            for bar, acc in zip(bars_purified, purified_accs):
                if acc > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{acc:.0f}',
                           ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Set labels and title
        ax.set_xlabel('Classifier', fontsize=18, fontweight='bold', labelpad=15)
        if def_idx == 0:
            ax.set_ylabel('Accuracy (%)', fontsize=18, fontweight='bold')
        ax.set_title(defense_name, fontsize=20, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(classifiers, fontsize=16, fontweight='bold')
        
        # Only show simplified legend in middle subplot
        if def_idx == 1:
            from matplotlib.patches import Patch
            
            legend_elements = [
                Patch(facecolor=test_colors['A'], edgecolor='black', label='Dataset A'),
                Patch(facecolor=test_colors['B'], edgecolor='black', label='Dataset B'),
                Patch(facecolor=test_colors['C'], edgecolor='black', label='Dataset C'),
                Patch(facecolor='gray', edgecolor='black', hatch='///', alpha=0.5, label='Defended'),
                Patch(facecolor='gray', edgecolor='black', alpha=0.9, label='Purified'),
            ]
            
            ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.18), 
                     ncol=5, fontsize=14, framealpha=0.9, edgecolor='black')
        
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.0)
        ax.set_ylim(0, 110)
        ax.tick_params(labelsize=16, width=1.5, length=6)
        
        # Add horizontal reference line
        ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    
    plt.suptitle('Cross-Dataset Generalization Performance Comprehensive Comparison', 
                fontsize=22, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comprehensive comparison chart saved to: {save_path}")
    
    plt.close()


def print_statistics(all_data):
    """Print statistical information"""
    print("\n" + "="*80)
    print("Cross-Dataset Experimental Results Statistics")
    print("="*80)
    
    for defense_name, data in all_data.items():
        print(f"\n【{defense_name} DEFENSE】")
        print("-" * 80)
        
        for test_ds in ['A', 'B', 'C']:
            test_key = f'test_{test_ds}'
            print(f"\nTest Dataset {test_ds}:")
            
            for clf in ['VarCNN', 'DF', 'LSTM', 'TMWF']:
                if clf in data[test_key]:
                    adv = data[test_key][clf]['adv']
                    purified = data[test_key][clf]['purified']
                    improvement = purified - adv
                    print(f"  {clf:12s}: Defended {adv:5.2f}% → Purified {purified:5.2f}% (Improvement {improvement:+5.2f}%)")


def main():
    """Main function"""
    print("\n" + "="*80)
    print("Cross-Dataset Performance Visualization Analysis - Version 2")
    print("Display adversarial accuracy and purified accuracy, use different styles to distinguish datasets")
    print("="*80)
    
    # Load data
    print("\nLoading complete test results...")
    all_data = load_complete_results()
    print("✓ Data loaded")
    
    # Print statistics
    print_statistics(all_data)
    
    # Create output directory
    output_dir = Path(__file__).parent / 'results' / 'cross_dataset_v2_en'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("Starting to generate visualization charts...")
    print("="*80)
    
    # 1. Plot individual defense method charts
    print("\n[1/4] Plotting Advtraffic cross-dataset performance...")
    plot_cross_dataset_performance('Advtraffic', all_data['Advtraffic'],
                                   save_path=output_dir / 'cross_dataset_Advtraffic_v2.png')
    
    print("\n[2/4] Plotting Walkie-Talkie cross-dataset performance...")
    plot_cross_dataset_performance('Walkie-Talkie', all_data['Walkie-Talkie'],
                                   save_path=output_dir / 'cross_dataset_Walkie-Talkie_v2.png')
    
    print("\n[3/4] Plotting Mockingbird cross-dataset performance...")
    plot_cross_dataset_performance('Mockingbird', all_data['Mockingbird'],
                                   save_path=output_dir / 'cross_dataset_Mockingbird_v2.png')
    
    # 2. Plot comprehensive comparison
    print("\n[4/4] Plotting comprehensive comparison...")
    plot_all_defenses_comparison(all_data,
                                 save_path=output_dir / 'cross_dataset_all_v2.png')
    
    print("\n" + "="*80)
    print("✓ All charts generated successfully!")
    print(f"✓ Charts saved to: {output_dir.absolute()}")
    print("="*80)
    print("\nGenerated chart files:")
    print("  1. cross_dataset_Advtraffic_v2.png - Advtraffic cross-dataset performance")
    print("  2. cross_dataset_Walkie-Talkie_v2.png - Walkie-Talkie cross-dataset performance")
    print("  3. cross_dataset_Mockingbird_v2.png - Mockingbird cross-dataset performance")
    print("  4. cross_dataset_all_v2.png - Comprehensive comparison")
    print("\nChart features:")
    print("  - Display adversarial accuracy and purified accuracy for each classifier")
    print("  - Use diagonal line fill for adversarial accuracy, solid fill for purified accuracy")
    print("  - Use different colors to distinguish test datasets (Blue=A, Red=B, Green=C)")
    print("\n")


if __name__ == '__main__':
    main()
