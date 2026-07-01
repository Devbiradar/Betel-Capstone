"""
fix_lowres.py
─────────────
Remove images below MIN_DIM threshold, replace with best quality
images from the discarded folder to maintain 2000 per class.
"""

import os
import shutil
import cv2
import imagehash
from PIL import Image
import numpy as np
import warnings

warnings.filterwarnings("ignore")

DATASET_PATH  = r"c:\Users\devub\Documents\Betel_capstone\Betel_capstone"
DISCARD_PATH  = r"c:\Users\devub\Documents\Betel_capstone_discarded"
MIN_DIM       = 100   # minimum acceptable width or height in pixels
HASH_THRESHOLD= 4

SKIP_DIRS = {"models", "checkpoints", "results", "__pycache__", "dataset"}


def get_quality_score(file_path):
    try:
        with Image.open(file_path) as img:
            img_rgb = img.convert("RGB")
            phash   = imagehash.phash(img_rgb)
            w, h    = img_rgb.size
            res     = w * h
        gray    = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        clarity = cv2.Laplacian(gray, cv2.CV_64F).var() if gray is not None else 0
        return phash, res, clarity
    except Exception:
        return None, 0, 0


def get_dimensions(file_path):
    try:
        with Image.open(file_path) as img:
            return img.size   # (width, height)
    except Exception:
        return (0, 0)


def main():
    classes = sorted([
        d for d in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, d)) and d not in SKIP_DIRS
    ])

    total_removed   = 0
    total_recovered = 0

    for cls in classes:
        cls_path     = os.path.join(DATASET_PATH, cls)
        discard_cls  = os.path.join(DISCARD_PATH,  cls)

        imgs = [f for f in os.listdir(cls_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]

        # ── Find low-res images ───────────────────────────────────────────────
        low_res = []
        for f in imgs:
            w, h = get_dimensions(os.path.join(cls_path, f))
            if min(w, h) < MIN_DIM:
                low_res.append(f)

        if not low_res:
            print(f"[{cls}] No low-res images found. OK")
            continue

        print(f"\n[{cls}] Found {len(low_res)} low-res image(s) to remove:")
        for f in low_res:
            w, h = get_dimensions(os.path.join(cls_path, f))
            path = os.path.join(cls_path, f)
            # Move to discarded folder
            os.makedirs(discard_cls, exist_ok=True)
            shutil.move(path, os.path.join(discard_cls, f))
            print(f"  Removed: {f}  ({w}x{h})")
            total_removed += 1

        n_to_recover = len(low_res)

        # ── Recover best replacements from discarded folder ───────────────────
        if not os.path.exists(discard_cls):
            print(f"  No discard folder for {cls} — cannot replace {n_to_recover} images.")
            continue

        discard_files = [
            os.path.join(discard_cls, f)
            for f in os.listdir(discard_cls)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

        # Score and filter: only keep those with min dim >= MIN_DIM
        candidates = []
        for fp in discard_files:
            w, h = get_dimensions(fp)
            if min(w, h) < MIN_DIM:
                continue
            phash, res, clarity = get_quality_score(fp)
            if phash is None:
                continue
            candidates.append({"path": fp, "hash": phash, "res": res, "clarity": clarity})

        if not candidates:
            print(f"  No valid replacements in discard folder for {cls}.")
            continue

        # Normalize and score
        clarities  = np.array([c["clarity"] for c in candidates])
        resolutions= np.array([c["res"]     for c in candidates])
        c_range = (clarities.max()   - clarities.min())   or 1
        r_range = (resolutions.max() - resolutions.min()) or 1
        for c in candidates:
            nc = (c["clarity"] - clarities.min()) / c_range
            nr = (c["res"]     - resolutions.min()) / r_range
            c["score"] = 0.7 * nc + 0.3 * nr

        candidates.sort(key=lambda x: x["score"], reverse=True)

        # Deduplicate
        unique = []
        for c in candidates:
            if not any(c["hash"] - u["hash"] <= HASH_THRESHOLD for u in unique):
                unique.append(c)

        to_recover = unique[:n_to_recover]
        if len(to_recover) < n_to_recover:
            print(f"  WARNING: Only {len(to_recover)} replacements available (need {n_to_recover}).")

        for c in to_recover:
            dest = os.path.join(cls_path, os.path.basename(c["path"]))
            if os.path.exists(dest):
                base, ext = os.path.splitext(os.path.basename(c["path"]))
                dest = os.path.join(cls_path, f"{base}_fix{ext}")
            shutil.move(c["path"], dest)
            w, h = get_dimensions(dest)
            print(f"  Replaced with: {os.path.basename(dest)}  ({w}x{h})")
            total_recovered += 1

    print(f"\n{'='*50}")
    print(f"Summary: Removed {total_removed} low-res images, Recovered {total_recovered} replacements.")

    # Final counts
    print("\nFinal image counts per class:")
    for cls in sorted(classes):
        cls_path = os.path.join(DATASET_PATH, cls)
        count = len([f for f in os.listdir(cls_path)
                     if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
        print(f"  {cls}: {count}")


if __name__ == "__main__":
    main()
