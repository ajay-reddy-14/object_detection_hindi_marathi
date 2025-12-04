# convert_coco_to_classification.py
import json, os, shutil, random
from PIL import Image

# === CONFIG ===
COCO_FILE = "dataset/test/_annotations.coco.json"   # path to your COCO file
IMAGE_DIR = "dataset/test"                          # folder that contains the images referenced in COCO
OUT_DIR = "new_dataset"                            # output dataset root
TRAIN_SPLIT = 0.8
VALID_SPLIT = 0.1
TEST_SPLIT = 0.1
# ==============

assert abs((TRAIN_SPLIT + VALID_SPLIT + TEST_SPLIT) - 1.0) < 1e-6, "splits must sum to 1.0"

# Load COCO
with open(COCO_FILE, "r") as f:
    coco = json.load(f)

images_map = {img['id']: img['file_name'] for img in coco.get('images', [])}
categories = {cat['id']: cat['name'] for cat in coco.get('categories', [])}

# Prepare output folders
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
for split in ("train","valid","test"):
    os.makedirs(os.path.join(OUT_DIR, split), exist_ok=True)
    for cname in categories.values():
        os.makedirs(os.path.join(OUT_DIR, split, cname), exist_ok=True)

# Group annotations by image to avoid many tiny crops from the same image being randomly placed with different split
anns_by_image = {}
for ann in coco.get('annotations', []):
    img_id = ann['image_id']
    anns_by_image.setdefault(img_id, []).append(ann)

# Process each image and its annotations
for img_id, anns in anns_by_image.items():
    img_name = images_map.get(img_id)
    if not img_name:
        continue
    img_path = os.path.join(IMAGE_DIR, img_name)
    if not os.path.exists(img_path):
        # try alternative path - sometimes COCO image paths are relative differently
        # skip if missing
        continue
    try:
        img = Image.open(img_path).convert("RGB")
    except:
        continue

    # decide split for this image (so all crops from same image go to same split)
    r = random.random()
    if r < TRAIN_SPLIT:
        split = "train"
    elif r < TRAIN_SPLIT + VALID_SPLIT:
        split = "valid"
    else:
        split = "test"

    for ann in anns:
        x, y, w, h = ann.get("bbox", [0,0,0,0])
        w, h = int(w), int(h)
        if w <= 0 or h <= 0:
            continue
        x1, y1 = int(max(0, x)), int(max(0, y))
        x2, y2 = int(min(img.width, x + w)), int(min(img.height, y + h))
        if x1 >= x2 or y1 >= y2:
            continue
        crop = img.crop((x1, y1, x2, y2))
        if crop.width == 0 or crop.height == 0:
            continue

        class_name = categories.get(ann['category_id'], "unknown")
        # sanitize class_name for folder safety
        safe_name = str(class_name).replace(" ", "_").replace("/", "_")
        out_path = os.path.join(OUT_DIR, split, safe_name, f"{img_id}_{ann['id']}.jpg")
        try:
            crop.save(out_path)
        except:
            # skip problematic writes
            continue

print("Conversion complete. Check folder:", OUT_DIR)
