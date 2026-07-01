import os
import cv2
import shutil
import imagehash
from PIL import Image
import numpy as np
import warnings

warnings.filterwarnings("ignore")

MAIN_DIR = r"c:\Users\devub\Documents\Betel_capstone\Betel_capstone"
DISCARD_DIR = r"c:\Users\devub\Documents\Betel_capstone_discarded"

# How many images to recover per class
RECOVER_CONFIG = {
    "Northern leaf blight": 300,
    "Gray spot": 200,
}

def get_image_metrics(file_path):
    try:
        with Image.open(file_path) as img:
            img_rgb = img.convert('RGB')
            phash = imagehash.phash(img_rgb)
            width, height = img_rgb.size
            resolution = width * height

        cv_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if cv_img is None:
            return None

        clarity = cv2.Laplacian(cv_img, cv2.CV_64F).var()

        return {
            'path': file_path,
            'hash': phash,
            'resolution': resolution,
            'clarity': clarity
        }
    except Exception as e:
        print(f"  Error reading {os.path.basename(file_path)}: {e}")
        return None

HASH_THRESHOLD = 4

def recover_best(class_name, n_recover):
    discard_class_dir = os.path.join(DISCARD_DIR, class_name)
    main_class_dir = os.path.join(MAIN_DIR, class_name)

    if not os.path.exists(discard_class_dir):
        print(f"[{class_name}] No discarded folder found. Skipping.")
        return

    discard_files = [
        os.path.join(discard_class_dir, f)
        for f in os.listdir(discard_class_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ]

    print(f"\n[{class_name}] Found {len(discard_files)} discarded images. Analyzing quality...")

    metrics_list = []
    for f in discard_files:
        m = get_image_metrics(f)
        if m is not None:
            metrics_list.append(m)

    if not metrics_list:
        print(f"[{class_name}] No readable images found in discard folder.")
        return

    # Normalize and score
    clarities = np.array([m['clarity'] for m in metrics_list])
    resolutions = np.array([m['resolution'] for m in metrics_list])

    c_range = (clarities.max() - clarities.min()) or 1
    r_range = (resolutions.max() - resolutions.min()) or 1

    for m in metrics_list:
        norm_c = (m['clarity'] - clarities.min()) / c_range
        norm_r = (m['resolution'] - resolutions.min()) / r_range
        m['score'] = norm_c * 0.7 + norm_r * 0.3

    # Sort best first
    metrics_list.sort(key=lambda x: x['score'], reverse=True)

    # Deduplicate against themselves
    unique = []
    for m in metrics_list:
        is_dup = any(m['hash'] - u['hash'] <= HASH_THRESHOLD for u in unique)
        if not is_dup:
            unique.append(m)

    print(f"[{class_name}] {len(unique)} unique quality images available after internal dedup.")

    # Pick top N
    to_recover = unique[:n_recover]

    if len(to_recover) < n_recover:
        print(f"[{class_name}] WARNING: Only {len(to_recover)} images available (requested {n_recover}).")

    # Move to main dir
    moved = 0
    for m in to_recover:
        dest = os.path.join(main_class_dir, os.path.basename(m['path']))
        # Avoid filename collision
        if os.path.exists(dest):
            base, ext = os.path.splitext(os.path.basename(m['path']))
            dest = os.path.join(main_class_dir, f"{base}_recovered{ext}")
        shutil.move(m['path'], dest)
        moved += 1

    print(f"[{class_name}] Moved {moved} images back to main dataset.")

def main():
    for class_name, n in RECOVER_CONFIG.items():
        recover_best(class_name, n)

    print("\n=== Final counts ===")
    for class_name in os.listdir(MAIN_DIR):
        class_path = os.path.join(MAIN_DIR, class_name)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            print(f"{class_name}: {count}")

if __name__ == "__main__":
    main()
