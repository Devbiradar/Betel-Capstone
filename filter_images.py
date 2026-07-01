import os
import cv2
import shutil
import imagehash
from PIL import Image
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings

# Suppress PIL warnings
warnings.filterwarnings("ignore")

DATA_DIR = r"c:\Users\devub\Documents\Betel_capstone\Betel_capstone"
DISCARD_DIR = r"c:\Users\devub\Documents\Betel_capstone_discarded"
MAX_IMAGES_PER_CLASS = 2000
HASH_THRESHOLD = 4  # hamming distance <= 4 is considered duplicate

def get_image_metrics(file_path):
    try:
        # Load for PIL / Hash
        with Image.open(file_path) as img:
            img = img.convert('RGB')
            phash = imagehash.phash(img)
            width, height = img.size
            resolution = width * height
            
        # Load for OpenCV / Clarity
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
        print(f"Error processing {file_path}: {e}")
        return None

def process_class(class_dir, discard_class_dir):
    print(f"\nProcessing class: {Path(class_dir).name}")
    image_files = [
        os.path.join(class_dir, f) for f in os.listdir(class_dir)
        if os.path.isfile(os.path.join(class_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ]
    
    print(f"Found {len(image_files)} images.")
    if not image_files:
        return
        
    metrics_list = []
    for f in image_files:
        m = get_image_metrics(f)
        if m is not None:
            metrics_list.append(m)
        else:
            # Move defective images directly
            os.makedirs(discard_class_dir, exist_ok=True)
            shutil.move(f, os.path.join(discard_class_dir, os.path.basename(f)))
            print(f"Moved defective image {os.path.basename(f)}")
            
    # 1. Deduplication (Group by similar hash)
    # Sort by quality first so we keep the best version when checking duplicates
    # Quality score for deduplication: normalize clarity and resolution
    if not metrics_list:
        return
        
    clarities = np.array([m['clarity'] for m in metrics_list])
    resolutions = np.array([m['resolution'] for m in metrics_list])
    
    c_min, c_max = clarities.min(), clarities.max()
    r_min, r_max = resolutions.min(), resolutions.max()
    
    c_range = (c_max - c_min) if c_max > c_min else 1
    r_range = (r_max - r_min) if r_max > r_min else 1
    
    for m in metrics_list:
        norm_c = (m['clarity'] - c_min) / c_range
        norm_r = (m['resolution'] - r_min) / r_range
        m['score'] = norm_c * 0.7 + norm_r * 0.3  # Prioritize clarity over resolution
        
    metrics_list.sort(key=lambda x: x['score'], reverse=True)
    
    unique_images = []
    discarded_duplicates = []
    
    print("Deduplicating...")
    for m in metrics_list:
        is_duplicate = False
        for u in unique_images:
            if m['hash'] - u['hash'] <= HASH_THRESHOLD:
                is_duplicate = True
                break
        
        if is_duplicate:
            discarded_duplicates.append(m)
        else:
            unique_images.append(m)
            
    print(f"Removed {len(discarded_duplicates)} duplicate/near-duplicate images.")
    
    # Move discarded duplicates
    if discarded_duplicates:
        os.makedirs(discard_class_dir, exist_ok=True)
        for m in discarded_duplicates:
            shutil.move(m['path'], os.path.join(discard_class_dir, os.path.basename(m['path'])))
            
    # 2. Keep top N
    if len(unique_images) > MAX_IMAGES_PER_CLASS:
        images_to_discard = unique_images[MAX_IMAGES_PER_CLASS:]
        print(f"Discarding {len(images_to_discard)} bottom-quality images to meet max limit...")
        os.makedirs(discard_class_dir, exist_ok=True)
        for m in images_to_discard:
            shutil.move(m['path'], os.path.join(discard_class_dir, os.path.basename(m['path'])))
    else:
        print(f"Class size after deduplication ({len(unique_images)}) is within limit.")

def main():
    if not os.path.exists(DISCARD_DIR):
        os.makedirs(DISCARD_DIR)
        
    for item in os.listdir(DATA_DIR):
        item_path = os.path.join(DATA_DIR, item)
        if os.path.isdir(item_path):
            discard_class_dir = os.path.join(DISCARD_DIR, item)
            process_class(item_path, discard_class_dir)
            
if __name__ == "__main__":
    main()
