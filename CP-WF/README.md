# CP-WF: A Universal Contrastive Purification Framework for Website Fingerprinting

## Overview

CP-WF is a plug-and-play adversarial purification framework for website fingerprinting (WF) attacks. It decouples the purification process from any specific downstream classifier by combining **supervised contrastive learning** with a **TransUNet-1D purification architecture**, enabling it to restore traffic fingerprints destroyed by adversarial defenses without requiring access to target model parameters.

**Key features:**
- Model-agnostic: works with DF, Var-CNN, TMWF, LSTM-WF and other classifiers
- Zero-shot generalization: effective on unseen website categories
- Handles three major defenses: AdvTraffic, Walkie-Talkie, Mockingbird

---

## Project Structure

```
CP-WF/
├── models/                     # Model architectures
│   ├── purifier.py             # TransUNet-1D purification network
│   ├── encoder.py              # Independent encoder (semantic anchor)
│   ├── classifiers/            # WF classifiers (DF, VarCNN, TMWF, LSTM-WF)
│   └── losses.py               # Triple loss function
│
├── training/                   # Training scripts
│   ├── pretrain_encoder.py     # Stage 1: encoder pre-training
│   ├── train_purifier.py       # Stage 2: purifier training
│   └── configs/                # Training configurations
│
├── evaluation/                 # Evaluation scripts
│   ├── evaluate.py             # Cross-classifier evaluation
│   ├── evaluate_cross_dataset.py  # Cross-dataset generalization
│   └── ablation/               # Ablation study scripts
│
├── visualization/              # Visualization scripts
│   ├── visualize_tsne.py       # t-SNE feature visualization
│   ├── visualize_pca.py        # PCA feature visualization
│   └── visualize_cross_dataset.py
│
├── data/                       # Data processing
│   ├── data_process.py         # Raw data preprocessing
│   └── split_datasets.py       # Dataset splitting (A/B/C)
│
└── README.md
```

---

## Requirements

```
Python >= 3.8
PyTorch >= 1.10
numpy
scikit-learn
matplotlib
tqdm
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Dataset

We use the **CW100** dataset ([Rimmer et al., 2017](https://arxiv.org/abs/1708.06376)), containing traffic data from 100 monitored website categories (~2,500 samples each).

Traffic sequences are converted to burst sequences of fixed length L=512. The dataset is split into train/val/test sets in a 7:1:2 ratio.

For cross-dataset experiments, the dataset is divided into three subsets:
- **Dataset A**: 40 common classes (0–39) + 20 exclusive classes (40–59)
- **Dataset B**: 40 common classes (0–39) + 20 exclusive classes (60–79)
- **Dataset C**: 40 common classes (0–39) + 20 exclusive classes (80–99)

---

## Quick Start

### Stage 1: Pre-train the Encoder

```bash
python training/pretrain_encoder.py \
    --data_path processed_data \
    --epochs 100 \
    --batch_size 256 \
    --save_path saved_models/pretrained_encoder.pth
```

### Stage 2: Train the Purifier

```bash
# Full dataset
python training/train_purifier.py \
    --defense mockingbird \
    --encoder_path saved_models/pretrained_encoder.pth \
    --epochs 40 \
    --batch_size 128

# Single subset (for cross-dataset experiments)
python training/train_purifier.py \
    --defense mockingbird \
    --dataset A \
    --encoder_path saved_models/pretrained_encoder.pth
```

### Evaluation

```bash
# Cross-classifier evaluation
python evaluation/evaluate.py \
    --model saved_models/purifier_mockingbird.pth \
    --defense mockingbird \
    --classifier DF

# Cross-dataset evaluation
python evaluation/evaluate_cross_dataset.py \
    --defense mockingbird \
    --train_dataset A
```

### Ablation Study

```bash
cd evaluation/ablation
run_ablation_mockingbird.bat   # Windows
```

---

