import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import json

DATASET_ROOT = "new_dataset"
TEST_DIR = os.path.join(DATASET_ROOT, "test")
EXPERIMENTS_DIR = "experiments"
IMAGE_SIZE_DEFAULT = 224
IMAGE_SIZE_INCEPTION = 299

def evaluate_model(model_name):
    print(f"Evaluating {model_name}...")
    model_dir = os.path.join(EXPERIMENTS_DIR, model_name)
    model_path = os.path.join(model_dir, "best_model.h5")
    
    if not os.path.exists(model_path):
        print(f"No checkpoint found for {model_name}")
        return

    # Determine image size
    img_size = IMAGE_SIZE_INCEPTION if "inception" in model_name.lower() else IMAGE_SIZE_DEFAULT
    
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")
        return

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(img_size, img_size),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    y_true = test_gen.classes
    y_pred_probs = model.predict(test_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    try:
        auc = roc_auc_score(tf.keras.utils.to_categorical(y_true), y_pred_probs, multi_class='ovr')
    except:
        auc = 0.0

    metrics = {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1,
        "AUC": auc
    }
    
    print(f"Metrics for {model_name}: {metrics}")
    pd.DataFrame([metrics]).to_csv(os.path.join(model_dir, "metrics.csv"), index=False)

def main():
    if not os.path.exists(EXPERIMENTS_DIR):
        print("No experiments directory.")
        return

    for model_name in os.listdir(EXPERIMENTS_DIR):
        if model_name == "yolov8": continue # Skip yolo folder if it exists but empty
        if os.path.isdir(os.path.join(EXPERIMENTS_DIR, model_name)):
            evaluate_model(model_name)

if __name__ == "__main__":
    main()
