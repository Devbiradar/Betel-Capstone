"""
dataloader.py
─────────────
Dataset class, augmentation pipelines, train/val/test split loader.
Handles class-imbalance automatically via WeightedRandomSampler.
"""

import os
import warnings
from collections import Counter

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

warnings.filterwarnings("ignore")

# ─── Dataset Class ────────────────────────────────────────────────────────────

class BetelDataset(Dataset):
    """
    Generic PyTorch dataset for a flat-folder image structure.

    Parameters
    ----------
    file_paths : list[str]  – absolute paths to image files.
    labels     : list[int]  – integer class labels.
    transform  : callable   – torchvision transform applied on __getitem__.
    """

    def __init__(self, file_paths: list, labels: list, transform=None):
        self.file_paths = file_paths
        self.labels     = labels
        self.transform  = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.file_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ─── Augmentation Pipelines ───────────────────────────────────────────────────

# ImageNet normalisation statistics
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

def get_transforms(phase: str = "train", image_size: int = 224) -> transforms.Compose:
    """
    Return the appropriate torchvision transform pipeline.

    Training  – heavy augmentation to combat overfitting.
    Val/Test  – deterministic resize + normalise only.
    """
    if phase == "train":
        return transforms.Compose([
            # Resize slightly larger, then random-crop to target size
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),

            # Geometric augmentations
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=30),

            # Photometric augmentations
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2
            ),

            # Random zoom (scale 0.8-1.2)
            transforms.RandomResizedCrop(
                image_size, scale=(0.8, 1.2), ratio=(0.9, 1.1)
            ),

            # Occasional Gaussian blur to simulate out-of-focus shots
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3
            ),

            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ])


# ─── Dataset Loader ───────────────────────────────────────────────────────────

def load_dataset(dataset_path: str, seed: int = 42):
    """
    Scan a folder-per-class directory and split into train / val / test.
    If 'train', 'val', 'test' folders exist in dataset_path, load them directly.

    Returns
    -------
    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names, class_to_idx
    """
    has_splits = all(os.path.isdir(os.path.join(dataset_path, s)) for s in ["train", "val", "test"])

    if has_splits:
        print("[Dataset] Found pre-split train/val/test directories.")
        train_dir = os.path.join(dataset_path, "train")
        val_dir = os.path.join(dataset_path, "val")
        test_dir = os.path.join(dataset_path, "test")
        
        class_names = sorted([
            d for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d)) and not d.startswith(".")
        ])
        class_to_idx = {cls: i for i, cls in enumerate(class_names)}
        print(f"[Dataset] Found {len(class_names)} classes: {class_names}")

        def load_split(split_dir):
            paths, labels = [], []
            for cls in class_names:
                cls_dir = os.path.join(split_dir, cls)
                if not os.path.isdir(cls_dir):
                    continue
                img_files = [f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
                for fname in img_files:
                    paths.append(os.path.join(cls_dir, fname))
                    labels.append(class_to_idx[cls])
            return paths, labels

        X_train, y_train = load_split(train_dir)
        X_val, y_val = load_split(val_dir)
        X_test, y_test = load_split(test_dir)
        
        print(f"[Dataset] Loaded → train: {len(X_train)} | val: {len(X_val)} | test: {len(X_test)}")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names, class_to_idx

    # Collect class names (sorted for determinism)
    class_names = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
        and not d.startswith(".")
        and d not in {"models", "checkpoints", "results", "dataset", "__pycache__", "train", "val", "test"}
    ])
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    print(f"[Dataset] Found {len(class_names)} classes: {class_names}")

    all_paths:  list[str] = []
    all_labels: list[int] = []

    for cls in class_names:
        cls_dir = os.path.join(dataset_path, cls)
        img_files = [
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        for fname in img_files:
            all_paths.append(os.path.join(cls_dir, fname))
            all_labels.append(class_to_idx[cls])
        print(f"  {cls}: {len(img_files)} images")

    print(f"[Dataset] Total images: {len(all_paths)}")

    if len(all_paths) == 0:
        raise ValueError(f"No images found in dataset_path: '{dataset_path}'. Please check directory structure.")

    # Stratified 70 / 15 / 15 split
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        all_paths, all_labels,
        test_size=0.30, stratify=all_labels, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=0.50, stratify=y_tmp, random_state=seed
    )

    print(
        f"[Dataset] Split → train: {len(X_train)} | "
        f"val: {len(X_val)} | test: {len(X_test)}"
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names, class_to_idx


# ─── DataLoader Factory ───────────────────────────────────────────────────────

def get_dataloaders(
    dataset_path:  str,
    batch_size:    int = 32,
    image_size:    int = 224,
    seed:          int = 42,
    num_workers:   int = 4,
):
    """
    Build PyTorch DataLoaders with:
      • Weighted sampling on train split to handle class imbalance.
      • Standard deterministic loaders for val and test.

    Returns
    -------
    train_loader, val_loader, test_loader, class_names, class_weights_tensor
    """
    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names, _ = \
        load_dataset(dataset_path, seed)

    n_classes = len(class_names)

    # ── Class-imbalance detection & weighted sampler ─────────────────────────
    counts        = Counter(y_train)
    class_weights = [1.0 / counts[i] for i in range(n_classes)]
    sample_weights = [class_weights[label] for label in y_train]

    # Check imbalance
    max_ratio = max(counts.values()) / min(counts.values())
    if max_ratio > 1.5:
        print(f"[Dataset] Imbalance ratio {max_ratio:.2f}x — enabling WeightedRandomSampler.")
    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True,
    )

    # Class weights tensor for weighted loss (normalised)
    cw_array = np.array(class_weights, dtype=np.float32)
    cw_array /= cw_array.sum()
    class_weights_tensor = torch.tensor(cw_array * n_classes, dtype=torch.float32)

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds = BetelDataset(X_train, y_train, get_transforms("train", image_size))
    val_ds   = BetelDataset(X_val,   y_val,   get_transforms("val",   image_size))
    test_ds  = BetelDataset(X_test,  y_test,  get_transforms("test",  image_size))

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        sampler     = sampler,       # replaces shuffle=True
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )

    return train_loader, val_loader, test_loader, class_names, class_weights_tensor
