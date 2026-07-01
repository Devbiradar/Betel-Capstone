"""
utils.py
────────
Shared utilities: seeding, early stopping, CSV logging,
GPU memory tracking, and a graceful KeyboardInterrupt handler.
"""

import os
import csv
import random
import signal
import torch
import numpy as np


# ─── Reproducibility ─────────────────────────────────────────────────────────

def seed_everything(seed: int = 42):
    """Seed all random-number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[Seed] All random states seeded to {seed}.")


# ─── Running Average Meter ────────────────────────────────────────────────────

class AverageMeter:
    """Tracks a running average of a scalar (e.g. loss or accuracy)."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


# ─── Early Stopping ───────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stop training when monitored metric has not improved for `patience` epochs.

    Parameters
    ----------
    patience : int   – epochs to wait before stopping.
    min_delta: float – minimum change to count as an improvement.
    mode     : str   – "min" for loss, "max" for accuracy.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.counter    = 0
        self.best_epoch = 0

    @property
    def should_stop(self) -> bool:
        return self.counter >= self.patience

    def __call__(self, value: float, epoch: int) -> bool:
        """
        Returns True if the value improved (caller should save checkpoint).
        Advances the stall counter otherwise.
        """
        improved = (
            (value < self.best_value - self.min_delta) if self.mode == "min"
            else (value > self.best_value + self.min_delta)
        )
        if improved:
            self.best_value = value
            self.counter    = 0
            self.best_epoch = epoch
            return True
        else:
            self.counter += 1
            return False


# ─── CSV Logger ───────────────────────────────────────────────────────────────

class CSVLogger:
    """
    Appends a row of metric values to a CSV file after every epoch.
    Creates or resumes the file automatically.
    """

    def __init__(self, filepath: str, fieldnames: list[str]):
        self.filepath   = filepath
        self.fieldnames = fieldnames
        self._init_file()

    def _init_file(self):
        exists = os.path.isfile(self.filepath)
        self._file   = open(self.filepath, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        if not exists:
            self._writer.writeheader()

    def log(self, row: dict):
        """Write one epoch row.  Missing keys are filled with ''."""
        full_row = {k: row.get(k, "") for k in self.fieldnames}
        self._writer.writerow(full_row)
        self._file.flush()

    def close(self):
        self._file.close()


# ─── GPU Memory Reporter ──────────────────────────────────────────────────────

def print_gpu_memory(epoch: int, every: int = 10):
    """Print GPU memory usage every `every` epochs (no-op if no CUDA)."""
    if epoch % every != 0:
        return
    if not torch.cuda.is_available():
        print(f"  [Epoch {epoch}] GPU: N/A (running on CPU)")
        return
    allocated  = torch.cuda.memory_allocated()  / 1024**3
    reserved   = torch.cuda.memory_reserved()   / 1024**3
    total      = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(
        f"  [Epoch {epoch}] GPU memory — "
        f"Allocated: {allocated:.2f} GB | "
        f"Reserved: {reserved:.2f} GB | "
        f"Total: {total:.2f} GB"
    )


# ─── Graceful Interrupt Handler ───────────────────────────────────────────────

class GracefulInterruptHandler:
    """
    Context manager that intercepts SIGINT / KeyboardInterrupt so the
    caller can save a checkpoint before exiting.

    Usage
    -----
    with GracefulInterruptHandler() as h:
        for epoch in ...:
            ...
            if h.interrupted:
                save_checkpoint(...)
                break
    """

    def __init__(self):
        self.interrupted = False
        self._original   = None

    def __enter__(self):
        self._original = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handler)
        return self

    def _handler(self, signum, frame):
        print("\n[Interrupt] Ctrl-C detected — finishing current batch then saving…")
        self.interrupted = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal(signal.SIGINT, self._original)
        # Swallow KeyboardInterrupt so we handle it ourselves
        return exc_type is KeyboardInterrupt


# ─── Pretty Table Printer ────────────────────────────────────────────────────

def print_comparison_table(results: dict):
    """
    Print a formatted comparison table of all model results.

    Parameters
    ----------
    results : dict  mapping model_name → {accuracy, precision, recall, f1, roc_auc}
    """
    header = f"{'Model':<35} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}"
    sep    = "─" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for name, m in results.items():
        print(
            f"{name:<35} "
            f"{m.get('accuracy',0)*100:>9.2f}% "
            f"{m.get('precision',0):>10.4f} "
            f"{m.get('recall',0):>10.4f} "
            f"{m.get('f1',0):>10.4f} "
            f"{m.get('roc_auc',0):>10.4f}"
        )
    print(sep)
