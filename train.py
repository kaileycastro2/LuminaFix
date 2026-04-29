import cv2
import glob
import os
import json
import numpy as np
from datetime import datetime
from src.transfers.nilut_transfer import NILUTTransfer

BASE = "test_images/training_data"
META_PATH = "models/nilut/meta.json"
transfer = NILUTTransfer()

def update_meta(model_name, sample_count):
    """Update meta.json with training info."""
    meta = {}
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)
    meta[model_name] = {
        "last_trained": datetime.now().isoformat(),
        "training_samples": sample_count,
        "epochs": 1500
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

# Ask user which training mode to use
print("=" * 50)
print("NILUT Model Training")
print("=" * 50)
print("\nChoose training mode:")
print("1. Train all models (5 per-style models + 1 universal model)")
print("2. Train only universal model")
print()

while True:
    choice = input("Enter your choice (1 or 2): ").strip()
    if choice in ["1", "2"]:
        break
    print("Invalid choice. Please enter 1 or 2.")

train_per_style = (choice == "1")
print()

# Helper to find images with multiple extensions
def find_images(directory):
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp', '*.tiff']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, ext)))
        files.extend(glob.glob(os.path.join(directory, ext.upper())))
    return sorted(list(set(files)))

# Load content images
content_dirs = [
    f"{BASE}/neon/input",
    f"{BASE}/portrait/input",
    f"{BASE}/landscape"
]
content = []
for d in content_dirs:
    for f in find_images(d):
        img = cv2.imread(f)
        if img is not None:
            content.append(img)

print(f"Loaded {len(content)} content images")

# Collect reference images for universal model
all_references = []

# Train per-style models (if selected)
ref_dir = f"{BASE}/reference"
if train_per_style:
    print("\n" + "=" * 50)
    print("Training per-style models...")
    print("=" * 50)

for style_folder in sorted(os.listdir(ref_dir)):
    style_path = os.path.join(ref_dir, style_folder)
    if os.path.isdir(style_path):
        # Use all images in the reference folder
        ref_images = find_images(style_path)
        refs = []
        for r in ref_images:
            img = cv2.imread(r)
            if img is not None:
                refs.append(img)

        # Train per-style model only if user chose option 1
        if train_per_style:
            # Clean name for model file
            model_name = style_folder.split(". ")[1].lower().replace("-", "_")
            save_path = f"models/nilut/{model_name}.pt"

            # Resize refs to same size and combine for training
            h, w = 512, 512
            refs_resized = [cv2.resize(r, (w, h)) for r in refs]
            combined_ref = np.vstack(refs_resized)

            print(f"\nTraining {model_name} with {len(refs)} reference images...")
            transfer.pretrain_on_reference(combined_ref, content, save_path)
            update_meta(model_name, len(content))
            print(f"Saved {save_path}")

        # Collect for universal model (always needed for universal training)
        all_references.extend(refs)

# Train universal model on all styles
print("\n" + "=" * 50)
print("Training universal model on all styles...")
print("=" * 50)
print(f"Using {len(all_references)} reference images from all style folders")

# Backup existing model if it exists
latest_dir = "models/nilut/latest"
model_path = os.path.join(latest_dir, "universal.pt")

if os.path.exists(model_path):
    # Get modification time of existing model
    mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
    timestamp_dir = f"models/nilut/universal/{mod_time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(timestamp_dir, exist_ok=True)

    # Move model to timestamped folder
    backup_path = os.path.join(timestamp_dir, "universal.pt")

    import shutil
    shutil.move(model_path, backup_path)
    print(f"Backed up previous model to: {backup_path}")

# Create latest directory if it doesn't exist
os.makedirs(latest_dir, exist_ok=True)

transfer.train_universal_model(
    reference_images=all_references,
    sample_images=content,
    save_path=model_path,
    epochs=1500
)
update_meta("universal", len(content))
print(f"Saved {model_path}")

print("\n" + "=" * 50)
print("Training complete!")
print("=" * 50)
