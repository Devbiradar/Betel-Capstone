"""
evaluate.py
───────────
Model evaluation helpers:
  • Confusion matrix  (plotted + saved)
  • Classification report  (printed + returned)
  • ROC-AUC curves  (one-vs-rest, multi-class)
  • Training / validation loss & accuracy curves
  • Full evaluation wrapper that runs inference and calls all of the above
"""

import os

import matplotlib
matplotlib.use("Agg")       # headless / VM-safe backend — no display needed
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm


# ─── Confusion Matrix ─────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true:      list,
    y_pred:      list,
    class_names: list[str],
    save_path:   str,
    model_name:  str = "",
):
    """Save a styled confusion matrix as a PNG file."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot      = True,
        fmt        = "d",
        cmap       = "Blues",
        xticklabels = class_names,
        yticklabels = class_names,
        ax         = ax,
        linewidths = 0.5,
        linecolor  = "grey",
    )
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label",      fontsize=13)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=15, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [Eval] Confusion matrix saved → {save_path}")


# ─── ROC-AUC Curves ───────────────────────────────────────────────────────────

def plot_roc_curves(
    y_true:      np.ndarray,   # shape [N]
    y_score:     np.ndarray,   # shape [N, n_classes]  (softmax probabilities)
    class_names: list[str],
    save_path:   str,
    model_name:  str = "",
):
    """Save per-class one-vs-rest ROC curves (+ macro average) as PNG."""
    n_classes = len(class_names)
    # Binarise ground truth
    y_onehot = np.zeros((len(y_true), n_classes), dtype=int)
    y_onehot[np.arange(len(y_true)), y_true] = 1

    fig, ax = plt.subplots(figsize=(10, 8))
    colors  = plt.cm.tab10(np.linspace(0, 1, n_classes))

    auc_scores: list[float] = []
    for i, (cls, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_onehot[:, i], y_score[:, i])
        roc_auc     = auc(fpr, tpr)
        auc_scores.append(roc_auc)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls}  (AUC = {roc_auc:.3f})")

    # Macro-average AUC
    macro_auc = np.mean(auc_scores)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate",  fontsize=13)
    ax.set_title(
        f"ROC Curves (one-vs-rest) — {model_name}\nMacro-AUC = {macro_auc:.4f}",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [Eval] ROC-AUC curves saved → {save_path}")
    return macro_auc


# ─── Training Curves ──────────────────────────────────────────────────────────

def plot_training_curves(history: dict, save_path: str, model_name: str = ""):
    """
    Plot loss and accuracy curves for train and validation.

    Parameters
    ----------
    history  : dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
                (each a list of per-epoch values)
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss",     color="steelblue",  lw=2)
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",       color="darkorange",  lw=2)
    axes[0].set_title(f"Loss — {model_name}", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train Accuracy", color="steelblue",  lw=2)
    axes[1].plot(epochs, history["val_acc"],   label="Val Accuracy",   color="darkorange",  lw=2)
    axes[1].set_title(f"Accuracy — {model_name}", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Training History — {model_name}", fontsize=15, y=1.01, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Eval] Training curves saved → {save_path}")


# ─── Full Evaluation Wrapper ──────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(
    model:       torch.nn.Module,
    dataloader,
    device:      torch.device,
    class_names: list[str],
    results_dir: str,
    model_name:  str,
) -> dict:
    """
    Run inference on a DataLoader and produce:
      • Classification report (printed)
      • Confusion matrix (saved)
      • ROC-AUC curves (saved)

    Returns a dict of summary metrics.
    """
    model.eval()
    all_labels:  list[int]   = []
    all_preds:   list[int]   = []
    all_probs:   list[list]  = []

    for imgs, labels in tqdm(dataloader, desc=f"  Evaluating {model_name}", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        probs  = F.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    y_true  = np.array(all_labels)
    y_pred  = np.array(all_preds)
    y_score = np.array(all_probs)

    # ── Classification report ─────────────────────────────────────────────────
    report = classification_report(
        y_true, y_pred,
        labels       = np.arange(len(class_names)),
        target_names = class_names,
        digits       = 4,
        zero_division= 0,
    )
    print(f"\n[{model_name}] Classification Report:\n{report}")

    # ── Summary scalars ───────────────────────────────────────────────────────
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall":    recall_score(   y_true, y_pred, average="macro", zero_division=0),
        "f1":        f1_score(       y_true, y_pred, average="macro", zero_division=0),
    }

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm_path = os.path.join(results_dir, f"{model_name}_confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path, model_name)

    # ── ROC-AUC curves ────────────────────────────────────────────────────────
    roc_path       = os.path.join(results_dir, f"{model_name}_roc_auc.png")
    macro_auc      = plot_roc_curves(y_true, y_score, class_names, roc_path, model_name)
    metrics["roc_auc"] = macro_auc

    return metrics


# ─── QSVM Evaluation (sklearn path) ──────────────────────────────────────────

def evaluate_qsvm(
    clf,             # fitted sklearn estimator
    X_test:     np.ndarray,
    y_test:     np.ndarray,
    K_test:     np.ndarray,   # precomputed kernel matrix [n_test × n_train]
    class_names: list[str],
    results_dir: str,
    model_name:  str = "QSVM",
) -> dict:
    """Evaluation path for the quantum kernel SVM (no DataLoader needed)."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    y_pred       = clf.predict(K_test)
    # decision_function scores for ROC-AUC (OvR)
    try:
        y_score  = clf.decision_function(K_test)   # [N, n_classes]
        if y_score.ndim == 1:                       # binary fallback
            y_score = np.column_stack([-y_score, y_score])
    except Exception:
        y_score = np.eye(len(class_names))[y_pred]  # one-hot fallback

    report = classification_report(y_test, y_pred, labels=np.arange(len(class_names)), target_names=class_names, digits=4, zero_division=0)
    print(f"\n[{model_name}] Classification Report:\n{report}")

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall":    recall_score(   y_test, y_pred, average="macro", zero_division=0),
        "f1":        f1_score(       y_test, y_pred, average="macro", zero_division=0),
    }

    cm_path  = os.path.join(results_dir, f"{model_name}_confusion_matrix.png")
    plot_confusion_matrix(y_test, y_pred, class_names, cm_path, model_name)

    try:
        roc_path       = os.path.join(results_dir, f"{model_name}_roc_auc.png")
        macro_auc      = plot_roc_curves(y_test, y_score, class_names, roc_path, model_name)
        metrics["roc_auc"] = macro_auc
    except Exception as e:
        print(f"  [QSVM] ROC-AUC skipped: {e}")
        metrics["roc_auc"] = 0.0

    return metrics
