from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from skimage import color
import os
import random
import json
import gc
import psutil

# -----------------------------
# Configuration
# -----------------------------
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32  # Reduced for better memory management
RANDOM_SEED = 42
STORAGE_LIMIT_GB = 60.0
MEMORY_LIMIT_PERCENT = 95  # Stop if memory usage exceeds this percentage
CHECK_INTERVAL = 500  # More frequent checks

# Convert to Path objects
raw_path = Path(RAW_DIR)
processed_path = Path(PROCESSED_DIR)

# Set random seed
random.seed(RANDOM_SEED)

def get_system_memory_usage():
    """Get current system memory usage percentage"""
    return psutil.virtual_memory().percent

def get_directory_size(path):
    """Get directory size in GB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total_size += os.path.getsize(fp)
            except:
                pass
    return total_size / (1024 ** 3)

def check_system_limits():
    """Check if we're approaching system limits"""
    storage_usage = get_directory_size(processed_path)
    memory_usage = get_system_memory_usage()
    
    storage_ok = storage_usage < STORAGE_LIMIT_GB
    memory_ok = memory_usage < MEMORY_LIMIT_PERCENT
    
    return storage_usage, memory_usage, storage_ok and memory_ok

def get_image_files_iterator(split_dir, max_files=None):
    """Get image files as an iterator with optional limit"""
    count = 0
    for img_path in split_dir.rglob("*.jpg"):
        if max_files and count >= max_files:
            return
        yield img_path
        count += 1

def save_batch_results(batch_results, split):
    """Save processed images from a batch with error handling"""
    successful_saves = 0
    for result in batch_results:
        if result is None:
            continue
            
        try:
            gray_out = processed_path / split / "grayscale" / result['name']
            color_out = processed_path / split / "color" / Path(result['name']).with_suffix(".npy").name
            
            # Create directories if they don't exist
            gray_out.parent.mkdir(parents=True, exist_ok=True)
            color_out.parent.mkdir(parents=True, exist_ok=True)
            
            # Save L channel as grayscale image
            L_save = ((result['L'] + 1.0) * 127.5).astype(np.uint8)
            Image.fromarray(L_save).save(gray_out)
            
            # Save AB channels as numpy array
            np.save(color_out, result['AB'])
            successful_saves += 1
            
        except Exception as e:
            print(f"Error saving {result['name']}: {e}")
            continue
    
    return successful_saves

def process_image(img_path):
    """Process a single image with comprehensive error handling"""
    try:
        # Open and resize with explicit close
        with Image.open(img_path) as img:
            img = img.convert("RGB").resize(IMAGE_SIZE)
        
        # Convert to numpy array [0, 255] -> [0, 1]
        img_rgb = np.array(img) / 255.0
        
        # Clear reference to help garbage collection
        del img
        
        # Convert RGB to CIELAB using skimage
        img_lab = color.rgb2lab(img_rgb)
        
        # Split channels
        L = img_lab[:, :, 0]      # L channel [0, 100]
        AB = img_lab[:, :, 1:]    # a, b channels [-127, 127]
        
        # Normalize to [-1, 1] range
        L_norm = (L / 50.0) - 1.0
        AB_norm = AB / 110.0
        
        # Clip to ensure strict [-1, 1] range
        L_norm = np.clip(L_norm, -1.0, 1.0)
        AB_norm = np.clip(AB_norm, -1.0, 1.0)
        
        return {
            'L': L_norm.astype(np.float32),
            'AB': AB_norm.astype(np.float32),
            'name': img_path.name,
            'path': str(img_path)
        }
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def process_split_sequential(split, max_files_per_split=None):
    """Process split with better resource management"""
    split_dir = raw_path / split
    if not split_dir.exists():
        print(f"⚠️ Skipping missing split folder: {split_dir}")
        return 0, True
    
    print(f"\n📂 Processing {split} split")
    
    processed_count = 0
    saved_count = 0
    continue_processing = True
    batch_results = []
    
    # Get total number of files for progress bar
    total_files = sum(1 for _ in split_dir.rglob("*.jpg"))
    if max_files_per_split:
        total_files = min(total_files, max_files_per_split)
    
    pbar = tqdm(total=total_files, desc=f"Processing {split}", unit="img")
    
    for img_path in get_image_files_iterator(split_dir, max_files_per_split):
        # Process image
        result = process_image(img_path)
        
        if result is not None:
            batch_results.append(result)
            processed_count += 1
        
        pbar.update(1)
        
        # Save batch when it reaches BATCH_SIZE
        if len(batch_results) >= BATCH_SIZE:
            saved_in_batch = save_batch_results(batch_results, split)
            saved_count += saved_in_batch
            batch_results.clear()
            
            # Force garbage collection
            gc.collect()
        
        # Check system limits periodically
        if processed_count % CHECK_INTERVAL == 0:
            storage_usage, memory_usage, limits_ok = check_system_limits()
            pbar.set_postfix({
                "Storage": f"{storage_usage:.2f}GB", 
                "Memory": f"{memory_usage:.1f}%",
                "Saved": f"{saved_count}/{processed_count}"
            })
            
            if not limits_ok:
                print(f"\n🚨 System limit reached! Storage: {storage_usage:.2f}GB, Memory: {memory_usage:.1f}%")
                continue_processing = False
                break
    
    # Save any remaining images in the last batch
    if batch_results:
        saved_in_batch = save_batch_results(batch_results, split)
        saved_count += saved_in_batch
        batch_results.clear()
        gc.collect()
    
    pbar.close()
    
    print(f"✅ {split}: Processed {processed_count}, Saved {saved_count}")
    return saved_count, continue_processing

# -----------------------------
# Main Processing
# -----------------------------
if not raw_path.exists():
    print(f"❌ Error: {raw_path} does not exist!")
    exit()

splits = [d.name for d in raw_path.iterdir() if d.is_dir()]
print(f"📁 Found splits: {splits}")

# Create output folders
for split in splits:
    for sub in ["grayscale", "color"]:
        (processed_path / split / sub).mkdir(parents=True, exist_ok=True)

# Process all splits
total_processed = 0
total_saved = 0
system_limits_reached = False

print("🎯 Processing dataset with system monitoring")
print(f"🛑 Storage limit: {STORAGE_LIMIT_GB} GB")
print(f"🛑 Memory limit: {MEMORY_LIMIT_PERCENT}%")
print("=" * 50)

# Optional: Limit files per split for testing
MAX_FILES_PER_SPLIT = 80000  # Set to a number like 5000 for testing

for split in splits:
    if system_limits_reached:
        print(f"⏹️  Skipping {split} due to system limits")
        continue
        
    saved_count, continue_processing = process_split_sequential(split, MAX_FILES_PER_SPLIT)
    total_saved += saved_count
    system_limits_reached = not continue_processing
    
    if system_limits_reached:
        break

# Final system check
storage_usage, memory_usage, _ = check_system_limits()
print(f"\n✅ Preprocessing complete!")
print(f"📊 Saved images: {total_saved:,} images")
print(f"💾 Storage used: {storage_usage:.2f} GB / {STORAGE_LIMIT_GB} GB")
print(f"🧠 Memory usage: {memory_usage:.1f}%")
print(f"💾 Processed data stored in: {processed_path}")

if system_limits_reached:
    print(f"🛑 Processing stopped due to system limits")
else:
    print(f"🎉 All available images processed successfully!")

# Validation
print("\n" + "=" * 50)
print("🔍 VALIDATION")
print("=" * 50)

for split in splits:
    grayscale_dir = processed_path / split / "grayscale"
    color_dir = processed_path / split / "color"
    
    if grayscale_dir.exists() and color_dir.exists():
        grayscale_files = len(list(grayscale_dir.glob("*.*")))
        color_files = len(list(color_dir.glob("*.npy")))
        
        print(f"📁 {split}:")
        print(f"   Grayscale images: {grayscale_files:,}")
        print(f"   Color .npy files: {color_files:,}")
        
        if grayscale_files == color_files:
            print(f"   ✅ Balanced: {grayscale_files:,} pairs")
        else:
            print(f"   ⚠️  Unbalanced! Difference: {abs(grayscale_files - color_files):,}")

# Create dataset metadata
print("\n" + "=" * 50)
print("📝 PROCESSING SUMMARY")
print("=" * 50)

summary = {
    "total_images_processed": total_saved,
    "storage_used_gb": round(storage_usage, 2),
    "memory_used_percent": round(memory_usage, 1),
    "splits_processed": splits,
    "stopped_due_to_limits": system_limits_reached
}

with open(processed_path / "processing-summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("📄 Created processing-summary.json")
print(f"💾 Data location: {processed_path}")