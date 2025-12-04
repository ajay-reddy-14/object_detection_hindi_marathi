import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# Settings
IMAGE_SIZE = 224
BATCH_SIZE = 32
DATASET_ROOT = "new_dataset"
VALID_DIR = os.path.join(DATASET_ROOT, "valid")
EXPERIMENTS_DIR = "experiments"

# Validation Generator (Common Dataset)
val_gen_obj = ImageDataGenerator(rescale=1./255)
val_gen = val_gen_obj.flow_from_directory(
    VALID_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

results = []

# Iterate over all model folders in experiments/
if not os.path.exists(EXPERIMENTS_DIR):
    print(f"No experiments found in {EXPERIMENTS_DIR}")
    exit()

model_folders = [f for f in os.listdir(EXPERIMENTS_DIR) if os.path.isdir(os.path.join(EXPERIMENTS_DIR, f))]

print(f"Found models: {model_folders}")

for model_name in model_folders:
    model_path = os.path.join(EXPERIMENTS_DIR, model_name, "model.h5")
    
    if not os.path.exists(model_path):
        print(f"Skipping {model_name}: model.h5 not found.")
        continue
        
    print(f"Evaluating {model_name}...")
    
    try:
        # Load Model
        model = tf.keras.models.load_model(model_path)
        
        # Predict
        val_gen.reset()
        preds = model.predict(val_gen, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        y_true = val_gen.classes
        
        # Metrics
        acc = accuracy_score(y_true, y_pred)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        try:
            # AUC (One-vs-Rest)
            # We must pass labels because y_true might not contain all classes
            all_labels = list(range(preds.shape[1]))
            auc = roc_auc_score(y_true, preds, multi_class='ovr', average='weighted', labels=all_labels)
        except Exception as e:
            print(f"Could not calculate AUC for {model_name}: {e}")
            auc = 0.0
            
        results.append({
            "Model": model_name,
            "Accuracy": acc,
            "AUC": auc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })
        
    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")

# Create DataFrame
df = pd.DataFrame(results)

# Display Table
print("\n" + "="*50)
print("FINAL MODEL COMPARISON")
print("="*50)
if not df.empty:
    # Sort by Accuracy
    df = df.sort_values(by="Accuracy", ascending=False)
    print(df.to_string(index=False))
    
    # Save to CSV
    df.to_csv("benchmark_results.csv", index=False)
    print("\nResults saved to benchmark_results.csv")
else:
    print("No results to display.")
