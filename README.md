# 🌿 Quantum-Classical Hybrid Betel Vine Disease Classifier

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?logo=pytorch&logoColor=white)
![PennyLane](https://img.shields.io/badge/PennyLane-0.35.0-6f42c1?logo=quantum&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research%20%7C%20Capstone-orange)

**A novel quantum-classical hybrid deep learning framework for automated betel vine (Piper betle) leaf disease classification using PennyLane variational quantum circuits integrated with EfficientNetB0 backbones.**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Disease Classes](#-disease-classes)
- [Architecture](#-architecture)
  - [Model 1 — Quantum Transfer Learning (QTL)](#model-1--quantum-transfer-learning-qtl)
  - [Model 2 — Quantum Attention Hybrid (QAH)](#model-2--quantum-attention-hybrid-qah)
  - [Model 3 — Quantum Kernel SVM (QSVM)](#model-3--quantum-kernel-svm-qsvm)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Dataset Setup](#-dataset-setup)
- [Usage](#-usage)
- [Training Details](#-training-details)
- [Evaluation Outputs](#-evaluation-outputs)
- [Requirements](#-requirements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔬 Overview

Betel vine (*Piper betle*) is a commercially important crop in South and Southeast Asia, widely used in cultural and medicinal contexts. Early and accurate detection of leaf diseases is critical to preventing significant agricultural losses.

This project proposes and evaluates **three quantum-classical hybrid models** for multi-class leaf disease classification:

| Model | Approach | Accuracy | ROC-AUC |
|-------|----------|----------|---------|
| **QAH** *(Main Proposed)* | EfficientNetB0 + Classical Attention Gate + Variational Quantum Circuit | **99.63%** | **0.9999** |
| **QTL** *(Proposed)* | EfficientNetB0 + Angle Embedding + Basic Entangler Layers | **89.64%** | **0.9879** |
| **QSVM** | EfficientNetB0 Features + PCA + ZZFeatureMap Kernel SVM | 40.83% | — |

> **Key Finding**: The Quantum Attention Hybrid (QAH) model achieves near-perfect classification, demonstrating that quantum variational circuits can effectively augment classical attention mechanisms for fine-grained agricultural image classification.

---

## 🌱 Disease Classes

The dataset contains **6 classes** of betel vine leaf conditions:

| # | Class | Description |
|---|-------|-------------|
| 1 | `Bacterial Leaf Disease` | Bacterial infection causing irregular lesions |
| 2 | `Dried Leaf` | Dried/desiccated leaves due to environmental stress |
| 3 | `Fungal Brown Spot Disease` | Fungal pathogen causing brown circular spots |
| 4 | `Healthy_Leaf` | Disease-free, healthy betel vine leaves |
| 5 | `Leaf_Rot` | Rotting caused by excess moisture or pathogens |
| 6 | `Leaf_Spot` | Spotting caused by various biotic or abiotic factors |

---

## 🏗️ Architecture

### Quantum Circuit Configuration

All quantum circuits use **PennyLane** with the `default.qubit` simulator and PyTorch backpropagation:

- **Qubits**: 6
- **Interface**: PyTorch (end-to-end differentiable)
- **Diff Method**: Backpropagation

---

### Model 1 — Quantum Transfer Learning (QTL)

```
Input Image (224×224×3)
        │
        ▼
EfficientNetB0 Backbone (frozen Phase 1 / unfrozen Phase 2)
        │  [B, 1280, 7, 7]  →  Global Avg Pool  →  [B, 1280]
        ▼
Pre-Quantum Compression: Linear(1280→256) → BN → GELU → Dropout → Linear(256→6) → Tanh
        │  [B, 6]  ∈ [-1, 1]
        ▼
Quantum Layer: AngleEmbedding(Y) + BasicEntanglerLayers (2 layers)
        │  [B, 6]  — Pauli-Z expectation values
        ▼
Classifier: Linear(6→64) → BN → GELU → Dropout → Linear(64→6)
        │
        ▼
Class Logits [B, 6]
```

**Quantum Circuit**:
- Encodes classical features as **Y-rotation angles** on 6 qubits
- Applies **2 layers of BasicEntanglerLayers** (parameterized + CNOT entanglement)
- Measures **⟨Z⟩** expectation values on all qubits

---

### Model 2 — Quantum Attention Hybrid (QAH)

> **This is the main proposed model achieving 99.63% accuracy.**

```
Input Image (224×224×3)
        │
        ▼
EfficientNetB0 Backbone  →  [B, 1280]
        │
        ├──────────────────────────────────┐
        │                                  │
        ▼                                  ▼
Classical SE-Attention Gate          Feature Stream
Linear(1280→320) → BN → GELU          (passthrough)
→ Linear(320→1280) → Sigmoid
        │  attention ∈ (0,1)
        └──── element-wise gate ──────────►│
                                           │  gated [B, 1280]
                                           ▼
                        Compress: Linear(1280→128) → BN → GELU → Dropout → Linear(128→6) → Tanh
                                           │  [B, 6]
                                           ▼
                        Quantum Attention:  AngleEmbedding(Y) + [RY+RZ+Ring CNOT] × 3 layers
                                           │  [B, 6]  quantum scores
                                           ▼
                        Fuse: concat([gated_feat, q_scores]) → [B, 1286]
                                           │
                                           ▼
                        Fusion Head: Linear(1286→512) → BN → GELU → Dropout
                                           │
                                           ▼
                        Classifier: Linear(512→128) → BN → GELU → Dropout → Linear(128→6)
                                           │
                                           ▼
                                   Class Logits [B, 6]
```

**Quantum Circuit** (3 variational layers):
- **AngleEmbedding (Y)** on attended/compressed features
- Per-layer: **RY + RZ** on each qubit + **Ring CNOT entanglement** (0→1→...→n-1→0)
- Measures **⟨Z⟩** on all 6 qubits as attention scores

---

### Model 3 — Quantum Kernel SVM (QSVM)

```
Images → Frozen EfficientNetB0 Feature Extractor → [N, 1280]
        │
        ▼
PCA (1280 → 6 dims)  +  MinMaxScaler ([0, π])
        │
        ▼
ZZFeatureMap Quantum Kernel:
  K[i,j] = |⟨ψ(x_i)|ψ(x_j)⟩|²
  (Hadamard + RZ + ZZ two-qubit phase interactions, 2 reps)
        │
        ▼
sklearn SVC(kernel='precomputed', C=1.0, decision='ovr')
        │
        ▼
6-Class Predictions
```

> **Note**: QSVM is used as a comparison baseline. Quantum kernel computation is O(N²) and runs on CPU simulation — practical only for subsampled datasets (≤350 train / ≤120 test).

---

## 📊 Results

### Model Performance Comparison

| Metric | QTL (Proposed) | QAH (Proposed) | QSVM |
|--------|:--------------:|:--------------:|:----:|
| **Accuracy** | 89.64% | **99.63%** | 40.83% |
| **Precision** (macro) | 0.8976 | **0.9963** | 0.2000 |
| **Recall** (macro) | 0.8963 | **0.9963** | 0.0817 |
| **F1** (macro) | 0.8954 | **0.9963** | 0.1160 |
| **ROC-AUC** (macro) | 0.9879 | **0.9999** | — |

### Result Visualizations

The following plots are generated automatically during training and saved to the `results/` directory:

| File | Description |
|------|-------------|
| `QAH_Proposed_training_curves.png` | Loss & accuracy curves (QAH) |
| `QAH_Proposed_confusion_matrix.png` | Per-class confusion matrix (QAH) |
| `QAH_Proposed_roc_auc.png` | One-vs-Rest ROC curves (QAH) |
| `QTL_Proposed_training_curves.png` | Loss & accuracy curves (QTL) |
| `QTL_Proposed_confusion_matrix.png` | Per-class confusion matrix (QTL) |
| `QTL_Proposed_roc_auc.png` | One-vs-Rest ROC curves (QTL) |
| `QSVM_confusion_matrix.png` | Confusion matrix (QSVM) |
| `QSVM_roc_auc.png` | ROC curves (QSVM) |
| `model_comparison.png` | Side-by-side bar chart of all models |
| `model_comparison.csv` | Tabular comparison of all metrics |

---

## 📁 Project Structure

```
Betel_capstone/
│
├── 📄 train.py                   # Main training script (entry point)
├── 📄 model.py                   # All 3 model architectures
├── 📄 dataloader.py              # Dataset class, augmentations, split loaders
├── 📄 evaluate.py                # Evaluation: confusion matrix, ROC-AUC, reports
├── 📄 utils.py                   # AverageMeter, EarlyStopping, CSVLogger, etc.
│
├── 📄 filter_images.py           # Utility: filter corrupt/low-quality images
├── 📄 fix_lowres.py              # Utility: upscale low-resolution images
├── 📄 topup_images.py            # Utility: augment to balance class sizes
├── 📄 recover_images.py          # Utility: recover mislabeled/misplaced images
├── 📄 generate_dashboards.py     # Generate HTML/PNG result dashboards
├── 📄 snapshot_dashboards.py     # Snapshot dashboard to file
├── 📄 plot_actual_results.py     # Plot final results from CSV
├── 📄 plot_inference.py          # Visualize inference on sample images
├── 📄 inference_time.py          # Benchmark model inference latency
│
├── 📄 requirements.txt           # Python dependencies
├── 📄 training_log.csv           # Per-epoch training metrics log
├── 📄 model_comparison.csv       # Final model comparison table
├── 📄 model_comparison.png       # Comparison bar chart
│
├── 📂 train/                     # Training images (per-class subdirectories)
│   ├── Bacterial Leaf Disease/
│   ├── Dried Leaf/
│   ├── Fungal Brown Spot Disease/
│   ├── Healthy_Leaf/
│   ├── Leaf_Rot/
│   └── Leaf_Spot/
├── 📂 val/                       # Validation images (same structure)
├── 📂 test/                      # Test images (same structure)
│
├── 📂 models/                    # (Optional) additional model files
├── 📂 results/                   # Auto-generated plots & metrics
│
├── 🔵 QAH_Proposed_best.pth      # Best QAH model checkpoint (~22 MB)
├── 🔵 QTL_Proposed_best.pth      # Best QTL model checkpoint (~17 MB)
│
└── 📄 .gitignore
```

---

## ⚙️ Installation

### Prerequisites

- Python **3.10** or higher
- `pip` package manager
- *(Optional but recommended)* CUDA-capable GPU for faster training

### 1. Clone the Repository

```bash
git clone https://github.com/Devbiradar/Betel-Capstone.git
cd Betel-Capstone
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **GPU Note**: If you have a CUDA GPU, install the CUDA-enabled PyTorch first:
> ```bash
> pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
> ```
> Then run `pip install -r requirements.txt` for the remaining packages.

---

## 📂 Dataset Setup

The dataset should follow this directory structure (pre-split format is supported):

```
Betel_capstone/
├── train/
│   ├── Bacterial Leaf Disease/   # .jpg / .jpeg / .png / .bmp images
│   ├── Dried Leaf/
│   ├── Fungal Brown Spot Disease/
│   ├── Healthy_Leaf/
│   ├── Leaf_Rot/
│   └── Leaf_Spot/
├── val/
│   └── (same 6 class folders)
└── test/
    └── (same 6 class folders)
```

**Alternatively**, if you have a flat folder-per-class dataset (no pre-split), the dataloader will automatically apply a **stratified 70/15/15 split**:

```
dataset_root/
├── Bacterial Leaf Disease/
├── Dried Leaf/
├── Fungal Brown Spot Disease/
├── Healthy_Leaf/
├── Leaf_Rot/
└── Leaf_Spot/
```

---

## 🚀 Usage

### Train All Models

```bash
# Train with default settings (50 epochs, batch size 32)
python train.py

# Specify custom dataset path
python train.py --dataset /path/to/your/dataset

# Custom hyperparameters
python train.py --epochs 100 --batch-size 16 --lr 5e-4

# Skip QSVM (very slow on CPU — recommended for quick experiments)
python train.py --skip-qsvm
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | Current directory | Path to dataset root folder |
| `--epochs` | `50` | Number of training epochs |
| `--batch-size` | `32` | Mini-batch size |
| `--lr` | `1e-3` | Initial learning rate |
| `--workers` | `4` | DataLoader worker processes |
| `--skip-qsvm` | `False` | Skip QSVM training (saves time) |

### Example — Quick Run (CPU-friendly)

```bash
python train.py --epochs 20 --batch-size 8 --skip-qsvm --workers 0
```

### Inference Time Benchmarking

```bash
python inference_time.py
```

### Plot Results from Existing CSV

```bash
python plot_actual_results.py
```

---

## 🏋️ Training Details

### Two-Phase Training Strategy

Both QTL and QAH use a **frozen-then-fine-tuned** training regime:

| Phase | Epochs | Backbone | Learning Rate |
|-------|--------|----------|---------------|
| **Phase 1** | 1–10 | ❄️ Frozen | `1e-3` (quantum + classifier only) |
| **Phase 2** | 11–50 | 🔓 Unfrozen | `1e-4` (full end-to-end) |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (weight decay `1e-4`) |
| Loss | CrossEntropyLoss + Label Smoothing (0.1) |
| LR Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Early Stopping | Patience = 10 epochs |
| Gradient Clipping | Max norm = 1.0 |
| Dropout | 0.3 |
| Image Size | 224 × 224 |
| Seed | 42 |

### Data Augmentation (Training)

| Transform | Value |
|-----------|-------|
| Random Crop | Resize to 256→crop 224 |
| Horizontal Flip | p = 0.5 |
| Vertical Flip | p = 0.3 |
| Random Rotation | ±30° |
| Color Jitter | brightness/contrast/saturation ±0.2 |
| Random Resized Crop | scale (0.8, 1.2) |
| Gaussian Blur | p = 0.3, σ ∈ (0.1, 2.0) |
| Normalize | ImageNet mean/std |

### Class Imbalance Handling

- **`WeightedRandomSampler`** is enabled automatically when class imbalance ratio > 1.5×
- **Weighted CrossEntropy Loss** with per-class weights inversely proportional to frequency

### Checkpoint Behavior

- Best model checkpoint is saved whenever `val_loss` improves
- Interrupted training (`Ctrl+C`) gracefully saves an `_interrupted.pth` checkpoint
- Re-running `train.py` automatically **skips training** if a `_best.pth` checkpoint exists and loads it directly

---

## 📈 Evaluation Outputs

After training, the following are automatically generated:

### Per-Model Outputs (in `results/`)
- **Confusion Matrix** — styled seaborn heatmap
- **ROC Curves** — one-vs-rest per class + macro AUC
- **Training Curves** — loss and accuracy vs. epoch

### Summary Outputs (project root)
- **`model_comparison.png`** — grouped bar chart across all 3 models
- **`model_comparison.csv`** — tabular metrics (accuracy, precision, recall, F1, ROC-AUC)
- **`training_log.csv`** — full per-epoch log for all models

---

## 📦 Requirements

```
torch==2.1.0
torchvision==0.16.0
pennylane==0.35.0
pennylane-lightning==0.35.0
scikit-learn==1.3.2
numpy==1.24.4
pandas==2.1.0
matplotlib==3.8.0
seaborn==0.13.0
tqdm==4.66.1
Pillow==10.1.0
opencv-python==4.8.1.78
```

---

## 🧪 Reproducing Results

All experiments use a fixed seed (`42`) for full reproducibility:

```python
seed_everything(42)   # seeds random, numpy, torch, cuda, PYTHONHASHSEED
```

The dataset split is **stratified 70/15/15** using `sklearn.model_selection.train_test_split` with `random_state=42`.

To reproduce the exact results from the paper:
1. Use the pre-split `train/`, `val/`, `test/` directories as provided
2. Run `python train.py` (QSVM results may vary slightly due to non-deterministic PennyLane sampling)

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **PennyLane** by Xanadu for quantum machine learning tooling
- **PyTorch** and **torchvision** for deep learning backbone
- **EfficientNet** (Tan & Le, 2019) for the pre-trained CNN architecture
- **scikit-learn** for classical ML components and metrics

---

<div align="center">

**Made with ❤️ as a Capstone Research Project**

*Quantum-Classical Hybrid AI for Sustainable Agriculture*

</div>
