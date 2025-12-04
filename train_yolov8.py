import os
import pandas as pd
import sys
print("Python Executable:", sys.executable)
print("Sys Path:", sys.path)
try:
    from ultralytics import YOLO
except ImportError as e:
    print("ImportError:", e)
    # Try to append site-packages manually if needed
    sys.path.append(r"C:\Users\ajayv\AppData\Local\Programs\Python\Python311\Lib\site-packages")
    from ultralytics import YOLO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import shutil

# Configuration
MODEL_NAME = "yolov8n-cls.pt" # Using nano model for speed, can change to yolov8s-cls.pt etc.
EPOCHS = 5
DATASET_ROOT = "new_dataset"
EXPERIMENT_DIR = os.path.join("experiments", "yolov8")

os.makedirs(EXPERIMENT_DIR, exist_ok=True)

def train_and_evaluate():
    print(f"=== Training YOLOv8 Classification ===")
    
    # YOLOv8 expects a specific dataset structure, which we already have (train/val/test folders)
    # We just need to point it to the root directory.
    
    # Load a model
    model = YOLO(MODEL_NAME)  # load a pretrained model (recommended for training)

    # Train the model
    # project=EXPERIMENT_DIR ensures results are saved there
    results = model.train(data=os.path.abspath(DATASET_ROOT), epochs=EPOCHS, imgsz=224, project=EXPERIMENT_DIR, name="train_run")

    # Validate/Test
    # YOLOv8 'val' mode usually uses the 'val' split. To test on 'test' split, we might need to trick it 
    # or just use the val metrics if the user is okay with that. 
    # But for strict comparison, we should evaluate on the 'test' set.
    # We can run validation on the 'test' folder by temporarily pointing 'val' to 'test' in a yaml? 
    # YOLO classification doesn't use a yaml file usually, it just uses directory structure.
    # It infers train/val/test from folder names.
    
    print("Evaluating on Test Set...")
    # We can use model.predict on the test folder and calculate metrics manually to match other models.
    
    test_dir = os.path.join(DATASET_ROOT, "test")
    classes = sorted(os.listdir(test_dir))
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    y_true = []
    y_pred = []
    y_probs = []
    
    for cls in classes:
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
            
        for img_file in os.listdir(cls_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(cls_dir, img_file)
            
            # Predict
            # verbose=False to reduce noise
            preds = model(img_path, verbose=False)
            
            # Extract result
            # preds[0].probs.top1 is the predicted class index
            # preds[0].probs.data is the probability distribution
            
            true_idx = class_to_idx[cls]
            pred_idx = preds[0].probs.top1
            probs = preds[0].probs.data.cpu().numpy()
            
            y_true.append(true_idx)
            y_pred.append(pred_idx)
            y_probs.append(probs)
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    try:
        # One-hot encode y_true for AUC
        y_true_onehot = np.zeros((y_true.size, y_true.max() + 1))
        y_true_onehot[np.arange(y_true.size), y_true] = 1
        # Ensure y_probs matches shape (sometimes classes might be missing in preds if model is weird, but usually fixed size)
        auc = roc_auc_score(y_true_onehot, y_probs, multi_class='ovr')
    except Exception as e:
        print(f"AUC calculation failed: {e}")
        auc = 0.0

    metrics = {
        "Model": "yolov8",
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1,
        "AUC": auc
    }
    
    print("Metrics:", metrics)
    pd.DataFrame([metrics]).to_csv(os.path.join(EXPERIMENT_DIR, "metrics.csv"), index=False)

if __name__ == "__main__":
    train_and_evaluate()
