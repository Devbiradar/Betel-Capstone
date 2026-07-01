"""
train.py
────────
Main training script for the Quantum-Classical Hybrid Betel Vine Disease
Classification pipeline.

Three models are trained and compared end-to-end:
  1. QuantumTransferLearning  (QTL  — proposed)
  2. QuantumAttentionHybrid   (QAH  — proposed)
  3. Quantum Kernel SVM       (QSVM — comparison)

Usage
-----
  python train.py [--dataset DATASET_PATH] [--epochs N] [--batch-size N]

Defaults
--------
  dataset    : same folder as train.py  (Betel_capstone structure)
  epochs     : 50
  batch-size : 32
"""

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")   # VM / headless safe
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from tqdm import tqdm

# ─── Local modules ────────────────────────────────────────────────────────────
from dataloader import get_dataloaders
from evaluate   import evaluate_model, evaluate_qsvm, plot_training_curves
from model      import (
    EfficientNetExtractor,
    QuantumAttentionHybrid,
    QuantumKernel,
    QuantumTransferLearning,
)
from utils import (
    AverageMeter,
    CSVLogger,
    EarlyStopping,
    GracefulInterruptHandler,
    print_comparison_table,
    print_gpu_memory,
    seed_everything,
)

# ─── Configuration ────────────────────────────────────────────────────────────

HERE         = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR     = os.path.join(HERE, "checkpoints")
RESULTS_DIR  = os.path.join(HERE, "results")
LOG_FILE     = os.path.join(HERE, "training_log.csv")

# Hyperparameters — override via CLI args
SEED         = 42
IMAGE_SIZE   = 224
BATCH_SIZE   = 32
N_EPOCHS     = 50
LR_INIT      = 1e-3
LR_FINETUNE  = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.1
DROPOUT      = 0.3
GRAD_CLIP    = 1.0
PATIENCE_ES  = 10     # early stopping patience
PATIENCE_LR  = 5      # ReduceLROnPlateau patience
FREEZE_EPOCHS= 10     # Phase 1: freeze backbone for first N epochs

# QSVM settings
N_QUBITS        = 6
QSVM_MAX_TRAIN  = 350   # subsample for quantum kernel (keeps it practical)
QSVM_MAX_TEST   = 120
QSVM_PCA_DIMS   = 6     # must equal N_QUBITS

# ─── Argument Parser ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Quantum-Classical Betel Vine Disease Classifier")
    p.add_argument("--dataset",    type=str, default=HERE,
                   help="Path to dataset root (folders = class names)")
    p.add_argument("--epochs",     type=int, default=N_EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr",         type=float, default=LR_INIT)
    p.add_argument("--workers",    type=int, default=4,
                   help="DataLoader worker processes")
    p.add_argument("--skip-qsvm", action="store_true",
                   help="Skip QSVM training (very slow on CPU)")
    return p.parse_args()


# ─── One Epoch: Training ──────────────────────────────────────────────────────

def train_one_epoch(
    model,
    loader,
    criterion:  nn.Module,
    optimizer:  torch.optim.Optimizer,
    device:     torch.device,
    grad_clip:  float = 1.0,
    scaler=None,
) -> tuple[float, float]:
    """
    Run one training epoch.
    Returns (avg_loss, accuracy_percent).
    """
    model.train()
    loss_meter = AverageMeter("loss")
    acc_meter  = AverageMeter("acc")

    pbar = tqdm(loader, desc="  Train", leave=False, unit="batch")
    for imgs, labels in pbar:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            # Mixed-precision forward pass
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs   = imgs.size(0)
        acc  = (logits.argmax(dim=1) == labels).float().mean().item() * 100
        loss_meter.update(loss.item(), bs)
        acc_meter .update(acc,         bs)
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.2f}%")

    return loss_meter.avg, acc_meter.avg


# ─── One Epoch: Validation ────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model,
    loader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float]:
    """Run one validation pass. Returns (avg_loss, accuracy_percent)."""
    model.eval()
    loss_meter = AverageMeter("loss")
    acc_meter  = AverageMeter("acc")

    for imgs, labels in tqdm(loader, desc="  Val  ", leave=False, unit="batch"):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss   = criterion(logits, labels)

        bs  = imgs.size(0)
        acc = (logits.argmax(dim=1) == labels).float().mean().item() * 100
        loss_meter.update(loss.item(), bs)
        acc_meter .update(acc,         bs)

    return loss_meter.avg, acc_meter.avg


# ─── Full Training Loop (Models 1 & 2) ───────────────────────────────────────

def train_model(
    tag:             str,
    model:           nn.Module,
    train_loader,
    val_loader,
    test_loader,
    class_names:     list[str],
    class_weights:   torch.Tensor,
    device:          torch.device,
    n_epochs:        int  = N_EPOCHS,
    lr:              float = LR_INIT,
    logger:          CSVLogger = None,
) -> dict:
    """
    Train one PyTorch model through two phases:
      Phase 1 (epochs 1–FREEZE_EPOCHS)  : backbone frozen, only quantum + classifier layers updated
      Phase 2 (epochs FREEZE_EPOCHS+1…) : full end-to-end fine-tuning at lower LR

    Returns summary metrics from test-set evaluation.
    """
    print(f"\n{'═'*60}")
    print(f"  Training: {tag}")
    print(f"{'═'*60}")

    model = model.to(device)

    # ── Check for existing checkpoint to skip training ────────────────────────
    best_ckpt = os.path.join(CKPT_DIR, f"{tag}_best.pth")
    if os.path.exists(best_ckpt):
        print(f"  [Info] Found existing checkpoint for {tag}. Skipping training!")
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt.get("state_dict", ckpt))
        print(f"  Loaded best checkpoint (val_acc={ckpt.get('val_acc', 0):.2f}%)")
        return evaluate_model(model, test_loader, device, class_names, RESULTS_DIR, tag)

    # ── Loss ──────────────────────────────────────────────────────────────────
    # Weighted CrossEntropy + label smoothing
    criterion = nn.CrossEntropyLoss(
        weight         = class_weights.to(device),
        label_smoothing= LABEL_SMOOTH,
    )

    # ── Optimizer / Scheduler (Phase 1) ───────────────────────────────────────
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = lr,
        weight_decay = WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=PATIENCE_LR
    )
    early_stop = EarlyStopping(patience=PATIENCE_ES, mode="min")

    # Mixed-precision scaler (no-op if CPU)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ── History ───────────────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_ckpt = os.path.join(CKPT_DIR, f"{tag}_best.pth")

    with GracefulInterruptHandler() as handler:
        for epoch in range(1, n_epochs + 1):

            # ── Phase transition at FREEZE_EPOCHS ─────────────────────────────
            if epoch == FREEZE_EPOCHS + 1:
                print(f"\n  [Phase 2] Unfreezing backbone — switching to LR={LR_FINETUNE}")
                model.unfreeze_backbone()
                # Rebuild optimizer with all parameters
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY
                )
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=PATIENCE_LR
                )

            t0 = time.time()
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, GRAD_CLIP, scaler
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            elapsed = time.time() - t0

            # LR schedule step on val loss
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            # History
            history["train_loss"].append(train_loss)
            history["val_loss"]  .append(val_loss)
            history["train_acc"] .append(train_acc)
            history["val_acc"]   .append(val_acc)

            # Console log
            print(
                f"  Epoch {epoch:03d}/{n_epochs}  |  "
                f"loss {train_loss:.4f}/{val_loss:.4f}  |  "
                f"acc {train_acc:.2f}/{val_acc:.2f}%  |  "
                f"lr {current_lr:.2e}  |  {elapsed:.1f}s"
            )

            # CSV log
            if logger:
                logger.log({
                    "model": tag, "epoch": epoch,
                    "train_loss": round(train_loss, 5),
                    "val_loss":   round(val_loss,   5),
                    "train_acc":  round(train_acc,  3),
                    "val_acc":    round(val_acc,    3),
                    "lr":         current_lr,
                })

            # GPU memory (every 10 epochs)
            print_gpu_memory(epoch, every=10)

            # Early stopping + checkpoint
            improved = early_stop(val_loss, epoch)
            if improved:
                torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                            "val_loss": val_loss, "val_acc": val_acc},
                           best_ckpt)
                print(f"  ✓ Checkpoint saved (val_acc={val_acc:.2f}%)")

            if early_stop.should_stop:
                print(f"\n  [EarlyStopping] No improvement for {PATIENCE_ES} epochs. Stopping.")
                break

            if handler.interrupted:
                print(f"\n  [Interrupt] Saving checkpoint and exiting…")
                torch.save({"epoch": epoch, "state_dict": model.state_dict()},
                           os.path.join(CKPT_DIR, f"{tag}_interrupted.pth"))
                break

    # ── Training curves ────────────────────────────────────────────────────────
    curve_path = os.path.join(RESULTS_DIR, f"{tag}_training_curves.png")
    plot_training_curves(history, curve_path, tag)

    # ── Load best weights then evaluate on test set ────────────────────────────
    if os.path.isfile(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"\n  Loaded best checkpoint (epoch={ckpt['epoch']}, val_acc={ckpt['val_acc']:.2f}%)")

    test_metrics = evaluate_model(
        model, test_loader, device, class_names, RESULTS_DIR, tag
    )
    return test_metrics


# ─── QSVM Pipeline ────────────────────────────────────────────────────────────

def train_qsvm(
    train_loader,
    test_loader,
    class_names: list[str],
    device:      torch.device,
) -> dict:
    """
    1. Extract features using frozen EfficientNetB0.
    2. PCA → N_QUBITS dimensions  (scaled to [0, π]).
    3. Subsample (practical quantum kernel computation).
    4. Build quantum ZZFeatureMap kernel matrix.
    5. Train sklearn SVC(kernel='precomputed').
    6. Evaluate on test subset.
    """
    tag = "QSVM"
    print(f"\n{'═'*60}")
    print(f"  Training: {tag}")
    print(f"{'═'*60}")

    # ── Step 1: Feature Extraction ────────────────────────────────────────────
    extractor = EfficientNetExtractor().to(device)
    extractor.eval()

    def extract(loader, max_samples=None, desc="Extracting"):
        feats, labels = [], []
        with torch.no_grad():
            for imgs, lbls in tqdm(loader, desc=f"  {desc}", leave=False):
                imgs = imgs.to(device, non_blocking=True)
                f    = extractor(imgs).cpu().numpy()
                feats.append(f)
                labels.extend(lbls.numpy().tolist())
                if max_samples and len(labels) >= max_samples:
                    break
        return np.vstack(feats)[:max_samples], np.array(labels[:max_samples])

    print(f"  Extracting CNN features (train subset ≤ {QSVM_MAX_TRAIN} samples)…")
    X_tr_raw, y_tr = extract(train_loader, QSVM_MAX_TRAIN, "Train features")
    print(f"  Extracting CNN features (test  subset ≤ {QSVM_MAX_TEST}  samples)…")
    X_te_raw, y_te = extract(test_loader,  QSVM_MAX_TEST,  "Test  features")

    # ── Step 2: PCA + Scale ───────────────────────────────────────────────────
    print(f"  Applying PCA({QSVM_PCA_DIMS}) + MinMaxScaler([0, π])…")
    pca    = PCA(n_components=QSVM_PCA_DIMS, random_state=SEED)
    scaler = MinMaxScaler(feature_range=(0, np.pi))

    X_tr = scaler.fit_transform(pca.fit_transform(X_tr_raw))
    X_te = scaler.transform(    pca.transform(    X_te_raw))

    print(f"  Explained variance ratio sum: {pca.explained_variance_ratio_.sum():.3f}")

    # ── Step 3: Quantum Kernel Matrices ───────────────────────────────────────
    qk = QuantumKernel(n_qubits=QSVM_PCA_DIMS)

    print(f"  Computing train kernel matrix ({len(X_tr)}×{len(X_tr)})…")
    print("  (This may take several minutes on CPU — quantum circuit simulation)")
    K_train = qk.kernel_matrix(X_tr, X_tr)

    print(f"  Computing test  kernel matrix ({len(X_te)}×{len(X_tr)})…")
    K_test  = qk.kernel_matrix(X_te, X_tr)

    # ── Step 4: SVC ───────────────────────────────────────────────────────────
    print("  Fitting SVC with precomputed quantum kernel…")
    clf = SVC(kernel="precomputed", C=1.0, decision_function_shape="ovr")
    clf.fit(K_train, y_tr)

    # ── Step 5: Evaluate ──────────────────────────────────────────────────────
    test_metrics = evaluate_qsvm(
        clf, X_te, y_te, K_test, class_names, RESULTS_DIR, tag
    )
    return test_metrics


# ─── Comparison Summary Plot ──────────────────────────────────────────────────

def save_comparison_plot(results: dict, class_names: list[str]):
    """Bar chart comparison of all 3 models on key metrics."""
    models  = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    labels  = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

    x   = np.arange(len(metrics))
    w   = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    colors  = ["steelblue", "darkorange", "seagreen"]

    for i, (mdl, color) in enumerate(zip(models, colors)):
        vals = [results[mdl].get(m, 0) for m in metrics]
        bars = ax.bar(x + i * w, vals, w, label=mdl, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x + w)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Quantum-Classical Hybrid Betel Vine Classifier",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n[Results] Comparison plot saved → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────────
    seed_everything(SEED)
    os.makedirs(CKPT_DIR,    exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Config] Device: {device}")
    if device.type == "cuda":
        print(f"         GPU   : {torch.cuda.get_device_name(0)}")
    print(f"         Dataset: {args.dataset}")
    print(f"         Epochs : {args.epochs}")
    print(f"         Batch  : {args.batch_size}")

    # ── DataLoaders ────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, class_names, class_weights = get_dataloaders(
        dataset_path = args.dataset,
        batch_size   = args.batch_size,
        image_size   = IMAGE_SIZE,
        seed         = SEED,
        num_workers  = args.workers,
    )
    print(f"[Config] Classes : {class_names}")

    # ── CSV Logger ────────────────────────────────────────────────────────────
    csv_fields = ["model", "epoch", "train_loss", "val_loss",
                  "train_acc", "val_acc", "lr"]
    logger = CSVLogger(LOG_FILE, csv_fields)

    # ─────────────────────────────────────────────────────────────────────────
    # MODEL 1: Quantum Transfer Learning (Proposed)
    # ─────────────────────────────────────────────────────────────────────────
    model1  = QuantumTransferLearning(
        n_classes = len(class_names),
        n_qubits  = N_QUBITS,
        n_layers  = 2,
        dropout   = DROPOUT,
    )
    res_qtl = train_model(
        tag            = "QTL_Proposed",
        model          = model1,
        train_loader   = train_loader,
        val_loader     = val_loader,
        test_loader    = test_loader,
        class_names    = class_names,
        class_weights  = class_weights,
        device         = device,
        n_epochs       = args.epochs,
        lr             = args.lr,
        logger         = logger,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # MODEL 2: Quantum Attention Hybrid (Proposed)
    # ─────────────────────────────────────────────────────────────────────────
    model2  = QuantumAttentionHybrid(
        n_classes = len(class_names),
        n_qubits  = N_QUBITS,
        n_layers  = 3,
        dropout   = DROPOUT,
    )
    res_qah = train_model(
        tag            = "QAH_Proposed",
        model          = model2,
        train_loader   = train_loader,
        val_loader     = val_loader,
        test_loader    = test_loader,
        class_names    = class_names,
        class_weights  = class_weights,
        device         = device,
        n_epochs       = args.epochs,
        lr             = args.lr,
        logger         = logger,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # MODEL 3: Quantum Kernel SVM
    # ─────────────────────────────────────────────────────────────────────────
    if not args.skip_qsvm:
        res_qsvm = train_qsvm(train_loader, test_loader, class_names, device)
    else:
        print("\n[QSVM] Skipped (--skip-qsvm flag set).")
        res_qsvm = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "roc_auc": 0}

    logger.close()

    # ─────────────────────────────────────────────────────────────────────────
    # COMPARISON
    # ─────────────────────────────────────────────────────────────────────────
    all_results = {
        "QTL (Proposed)":  res_qtl,
        "QAH (Proposed)":  res_qah,
        "QSVM":            res_qsvm,
    }
    print_comparison_table(all_results)
    save_comparison_plot(all_results, class_names)

    # Save comparison table to CSV
    rows = [{"Model": k, **v} for k, v in all_results.items()]
    df   = pd.DataFrame(rows)
    comp_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    df.to_csv(comp_path, index=False)
    print(f"[Results] Comparison table saved → {comp_path}")

    print("\n[Done] Training complete. All results saved to ./results/")


if __name__ == "__main__":
    main()
