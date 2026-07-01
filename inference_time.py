"""
inference_time.py
─────────────────
Measures inference latency for QTL_Baseline and QAH_Proposed models.

Reports:
  • Single-image inference time  (ms)
  • Batch  inference time        (ms/image)
  • FPS (frames per second)

Usage (on VM):
  python inference_time.py

Checkpoints are loaded from:  ./checkpoints/QTL_Baseline_best.pth
                               ./checkpoints/QAH_Proposed_best.pth
"""

import os
import time

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from model import QuantumTransferLearning, QuantumAttentionHybrid

# ─── Config ───────────────────────────────────────────────────────────────────
HERE        = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR    = HERE

N_CLASSES   = 6
N_QUBITS    = 6
IMAGE_SIZE  = 224
WARMUP_RUNS = 10      # GPU warmup iterations (critical for accurate GPU timing)
TIMED_RUNS  = 100     # Number of runs to average over
BATCH_SIZE  = 32      # For batch inference test

# ImageNet normalisation
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

# ─── Transform ────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])

# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_model(model_class, ckpt_name, **kwargs):
    """Load a model from checkpoint."""
    ckpt_path = os.path.join(CKPT_DIR, ckpt_name)
    model = model_class(**kwargs)
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        print(f"  ✓ Loaded: {ckpt_name}  (epoch={ckpt.get('epoch','?')}, val_acc={ckpt.get('val_acc', 0):.2f}%)")
    else:
        print(f"  ⚠ Checkpoint not found: {ckpt_path} — using random weights")
    model.eval()
    return model


def make_dummy_image(device):
    """Create a random single image tensor [1, 3, H, W]."""
    return torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)


def make_dummy_batch(device, batch_size=BATCH_SIZE):
    """Create a random batch tensor [B, 3, H, W]."""
    return torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)


def measure_latency(model, input_tensor, device, n_warmup=WARMUP_RUNS, n_runs=TIMED_RUNS):
    """
    Accurately measures inference latency.

    - Uses CUDA events for GPU timing (most accurate).
    - Falls back to time.perf_counter() for CPU.
    - Performs warmup runs before timing to avoid cold-start bias.
    """
    model = model.to(device)

    with torch.no_grad():
        # ── Warmup ─────────────────────────────────────────────────────────
        for _ in range(n_warmup):
            _ = model(input_tensor)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # ── Timed runs ─────────────────────────────────────────────────────
        latencies = []

        if device.type == "cuda":
            for _ in range(n_runs):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event   = torch.cuda.Event(enable_timing=True)
                start_event.record()
                _ = model(input_tensor)
                end_event.record()
                torch.cuda.synchronize()
                latencies.append(start_event.elapsed_time(end_event))  # ms
        else:
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = model(input_tensor)
                latencies.append((time.perf_counter() - t0) * 1000)    # ms

    return np.array(latencies)


def print_stats(name, latencies_ms, batch_size=1):
    """Pretty-print latency statistics."""
    per_image = latencies_ms / batch_size
    print(f"\n  {'─'*50}")
    print(f"  Model : {name}")
    print(f"  {'─'*50}")
    print(f"  Batch size           : {batch_size}")
    print(f"  Runs averaged        : {len(latencies_ms)}")
    print(f"  Mean  latency/image  : {per_image.mean():.3f} ms")
    print(f"  Std   latency/image  : {per_image.std():.3f} ms")
    print(f"  Min   latency/image  : {per_image.min():.3f} ms")
    print(f"  Max   latency/image  : {per_image.max():.3f} ms")
    print(f"  P95   latency/image  : {np.percentile(per_image, 95):.3f} ms")
    print(f"  FPS (1/mean_ms*1000) : {1000/per_image.mean():.1f} images/sec")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Inference Timing] Device : {device}")
    if device.type == "cuda":
        print(f"[Inference Timing] GPU    : {torch.cuda.get_device_name(0)}")

    print(f"[Inference Timing] Warmup runs : {WARMUP_RUNS}")
    print(f"[Inference Timing] Timed  runs : {TIMED_RUNS}")

    # ── Load models ──────────────────────────────────────────────────────────
    print("\n[Loading Models]")
    qtl = load_model(
        QuantumTransferLearning, "QTL_Baseline_best.pth",
        n_classes=N_CLASSES, n_qubits=N_QUBITS, n_layers=2, dropout=0.3,
    )
    qah = load_model(
        QuantumAttentionHybrid,  "QAH_Proposed_best.pth",
        n_classes=N_CLASSES, n_qubits=N_QUBITS, n_layers=3, dropout=0.3,
    )

    models = {
        "QTL_Baseline": qtl,
        "QAH_Proposed": qah,
    }

    # ── Single-image inference ────────────────────────────────────────────────
    print("\n" + "═"*55)
    print("  SINGLE-IMAGE INFERENCE (batch_size=1)")
    print("═"*55)

    single_img = make_dummy_image(device)

    for name, model in models.items():
        lat = measure_latency(model, single_img, device)
        print_stats(name, lat, batch_size=1)

    # ── Batch inference ───────────────────────────────────────────────────────
    print("\n" + "═"*55)
    print(f"  BATCH INFERENCE (batch_size={BATCH_SIZE})")
    print("═"*55)

    batch_imgs = make_dummy_batch(device, BATCH_SIZE)

    for name, model in models.items():
        lat = measure_latency(model, batch_imgs, device)
        print_stats(name, lat, batch_size=BATCH_SIZE)

    # ── Summary Table ─────────────────────────────────────────────────────────
    print("\n" + "═"*55)
    print("  SUMMARY")
    print("═"*55)
    print(f"  {'Model':<20} {'Single (ms)':<15} {'Batch/img (ms)':<15} {'FPS'}")
    print(f"  {'─'*53}")

    for name, model in models.items():
        s = measure_latency(model, make_dummy_image(device),   device).mean()
        b = measure_latency(model, make_dummy_batch(device),   device).mean() / BATCH_SIZE
        fps = 1000 / s
        print(f"  {name:<20} {s:<15.3f} {b:<15.3f} {fps:.1f}")

    print()


if __name__ == "__main__":
    main()
