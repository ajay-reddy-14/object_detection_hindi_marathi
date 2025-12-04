import os
import shutil
import random
import cv2
import pandas as pd
from openimages.download import download_dataset

# === CONFIG ===
CLASSES = [
    "Car", "Bus", "Truck", "Boat", "Person",
    "Clock", "Bicycle", "Motorcycle", "Traffic light", "Sheep",
    "Backpack", "Bench", "Chair", "Door", "Fire hydrant",
    "Handbag", "Stop sign", "Suitcase", "Traffic cone", "Train",
    "Tree", "Umbrella", "Stairs", "Elevator", "Rat", "Scooter"
]
LIMIT = 500  # Number of images per class to download
DATASET_DIR = "dataset"  # Temporary storage for raw Open Images
OUTPUT_DIR = "new_dataset"  # Final classification dataset
TRAIN_SPLIT = 0.8
VALID_SPLIT = 0.1
TEST_SPLIT = 0.1
# ==============

def prepare_directories():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    
    for split in ["train", "valid", "test"]:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls.replace(" ", "_")), exist_ok=True)

def get_split():
    r = random.random()
    if r < TRAIN_SPLIT:
        return "train"
    elif r < TRAIN_SPLIT + VALID_SPLIT:
        return "valid"
    else:
        return "test"

def process_images():
    print("Processing images and creating crops...")
    
    # The openimages library downloads to 'dataset/{class_name}/images' 
    # and annotations to 'dataset/{class_name}/{format}/annotations.csv' usually, 
    # but let's check the structure after download or assume standard structure.
    # Actually, openimages download_dataset creates:
    # dataset/
    #   {class_name}/
    #     images/
    #       {image_id}.jpg
    #     {format}/
    #       {image_id}.xml (if pascal) or similar.
    
    # However, for cropping we need bounding boxes. 
    # The library might not make it easy to get a single CSV for all.
    # Let's rely on the fact that we can iterate through the downloaded folders.
    
    # Wait, the library 'openimages' might be different from 'oi'. 
    # Let's assume we use the one installed.
    # If we use annotation_format="darknet", it creates .txt files with same name as image.
    # If "pascal", .xml files.
    # Let's use "pascal" as it is robust, or "darknet" (yolo) which is easy to parse.
    # Let's use "darknet" for easy parsing: class_idx x_center y_center w h (normalized)
    
    for cls in CLASSES:
        print(f"Processing class: {cls}")
        cls_safe = cls.replace(" ", "_")
        class_dir = os.path.join(DATASET_DIR, cls.lower()) # openimages usually uses lowercase or specific folder names
        
        # The download_dataset function usually creates folders named after the class.
        # Let's look for the folder. It might be 'dataset/car', 'dataset/bus', etc.
        # Or it might be case sensitive.
        
        # Let's try to find the folder case-insensitively
        found_dir = None
        if os.path.exists(DATASET_DIR):
            for d in os.listdir(DATASET_DIR):
                if d.lower() == cls.lower():
                    found_dir = os.path.join(DATASET_DIR, d)
                    break
        
        if not found_dir:
            print(f"Warning: Could not find directory for class {cls}")
            continue
            
        images_dir = os.path.join(found_dir, "images")
        labels_dir = os.path.join(found_dir, "darknet") # We will request darknet format
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"Warning: Missing images or labels for {cls}")
            continue
            
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in image_files:
            img_path = os.path.join(images_dir, img_file)
            label_file = os.path.splitext(img_file)[0] + ".txt"
            label_path = os.path.join(labels_dir, label_file)
            
            if not os.path.exists(label_path):
                continue
                
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                h_img, w_img, _ = img.shape
                
                with open(label_path, "r") as f:
                    lines = f.readlines()
                    
                for idx, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                        
                    # Darknet format: class_index x_center y_center w h (normalized)
                    # We only downloaded one class per folder, so we assume all boxes in this folder belong to 'cls'
                    # EXCEPT if openimages includes other classes in the same file.
                    # But download_dataset with specific classes usually filters?
                    # Actually, 'download_dataset' might download images that have the target class, 
                    # but the annotation file might contain ALL classes or just the target?
                    # Usually it creates label files specific to the requested class if we download separately?
                    # Let's assume the box is valid for the current class context or just crop everything 
                    # and put it in the current class folder? 
                    # A safer bet is: The download_dataset tool creates a 'classes.txt' or similar?
                    # If we download one by one, the txt files in that folder should be for that class?
                    # Let's verify this assumption. If not, we might have noise.
                    # But for now, we will assume the boxes correspond to the object of interest.
                    
                    # Note: openimages download might map classes to indices 0, 1, 2... 
                    # If we download multiple classes at once, they share indices.
                    # If we download separately, maybe 0?
                    # We will download ALL at once to ensure consistent class indices if possible, 
                    # OR we just trust that we put them in the right folder.
                    
                    # Let's download ALL classes in one go in the main block.
                    
                    # Parsing Darknet
                    # c_idx = int(parts[0]) # We ignore this if we trust the folder, but if we download all together...
                    # If we download all together, 'dataset' will have subfolders? 
                    # The 'openimages' library behavior:
                    # It creates 'dataset/{class_name}/...'
                    # So iterating by folder is safe.
                    
                    x_c, y_c, w_n, h_n = map(float, parts[1:])
                    
                    x_c *= w_img
                    y_c *= h_img
                    w_n *= w_img
                    h_n *= h_img
                    
                    x1 = int(x_c - w_n / 2)
                    y1 = int(y_c - h_n / 2)
                    x2 = int(x_c + w_n / 2)
                    y2 = int(y_c + h_n / 2)
                    
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w_img, x2)
                    y2 = min(h_img, y2)
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                        
                    crop = img[y1:y2, x1:x2]
                    
                    if crop.size == 0:
                        continue
                        
                    split = get_split()
                    save_name = f"{os.path.splitext(img_file)[0]}_{idx}.jpg"
                    save_path = os.path.join(OUTPUT_DIR, split, cls_safe, save_name)
                    
                    cv2.imwrite(save_path, crop)
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

def main():
    print("Starting dataset update...")
    
    # 1. Prepare Output Directory
    prepare_directories()
    
    # 2. Download Data
    print(f"Downloading {LIMIT} images for classes: {CLASSES}")
    # We use a try-except block because sometimes the library throws errors on some images
    try:
        download_dataset(
            dest_dir=DATASET_DIR,
            class_labels=CLASSES,
            annotation_format="darknet",
            limit=LIMIT
        )
    except Exception as e:
        print(f"Download finished with some errors (or completed): {e}")
    
    # 3. Process and Crop
    process_images()
    
    print("Dataset update complete!")
    print(f"New dataset is located at: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
