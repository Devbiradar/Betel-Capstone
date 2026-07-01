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
TARGET = 2000

HASH_THRESHOLD = 4

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

def recover_to_target(class_name, current_count):
    n_recover = TARGET - current_count
    if n_recover <= 0:
        print(f"[{class_name}] Already at or above target ({current_count}). Skipping.")
        return

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

    print(f"\n[{class_name}] Current: {current_count} | Need: {n_recover} more | Available in discard: {len(discard_files)}")

    metrics_list = []
    for f in discard_files:
        m = get_image_metrics(f)
        if m is not None:
            metrics_list.append(m)

    if not metrics_list:
        print(f"[{class_name}] No readable images in discard folder.")
        return

    # Score and sort
    clarities = np.array([m['clarity'] for m in metrics_list])
    resolutions = np.array([m['resolution'] for m in metrics_list])

    c_range = (clarities.max() - clarities.min()) or 1
    r_range = (resolutions.max() - resolutions.min()) or 1

    for m in metrics_list:
        norm_c = (m['clarity'] - clarities.min()) / c_range
        norm_r = (m['resolution'] - resolutions.min()) / r_range
        m['score'] = norm_c * 0.7 + norm_r * 0.3

    metrics_list.sort(key=lambda x: x['score'], reverse=True)

    # Deduplicate
    unique = []
    for m in metrics_list:
        is_dup = any(m['hash'] - u['hash'] <= HASH_THRESHOLD for u in unique)
        if not is_dup:
            unique.append(m)

    print(f"[{class_name}] {len(unique)} unique images available after dedup.")

    to_recover = unique[:n_recover]

    if len(to_recover) < n_recover:
        print(f"[{class_name}] WARNING: Only {len(to_recover)} available (needed {n_recover}).")

    moved = 0
    for m in to_recover:
        dest = os.path.join(main_class_dir, os.path.basename(m['path']))
        if os.path.exists(dest):
            base, ext = os.path.splitext(os.path.basename(m['path']))
            dest = os.path.join(main_class_dir, f"{base}_recovered{ext}")
        shutil.move(m['path'], dest)
        moved += 1

    print(f"[{class_name}] Moved {moved} images back. New count: {current_count + moved}")

def main():
    # Get current counts
    current_counts = {}
    for item in os.listdir(MAIN_DIR):
        item_path = os.path.join(MAIN_DIR, item)
        if os.path.isdir(item_path):
            count = len([
                f for f in os.listdir(item_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ])
            current_counts[item] = count

    print("Current counts:")
    for cls, cnt in current_counts.items():
        print(f"  {cls}: {cnt}")

    # Only recover for classes below target
    for class_name, count in current_counts.items():
        if count < TARGET:
            recover_to_target(class_name, count)

    print("\n=== Final counts ===")
    for item in os.listdir(MAIN_DIR):
        item_path = os.path.join(MAIN_DIR, item)
        if os.path.isdir(item_path):
            count = len([
                f for f in os.listdir(item_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ])
            print(f"{item}: {count}")

if __name__ == "__main__":
    main()
