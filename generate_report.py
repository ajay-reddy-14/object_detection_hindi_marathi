import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model_factory import get_model

# Configuration
MODELS = ["mobilenetv2", "resnet50", "vgg16", "efficientnetb0", "densenet121", "inceptionv3"]
DATASET_ROOT = "new_dataset"
TEST_DIR = os.path.join(DATASET_ROOT, "test")
IMAGE_SIZE = 224
BATCH_SIZE = 32

from sklearn.metrics import ConfusionMatrixDisplay

def plot_confusion_matrix(cm, classes, save_path):
    try:
        fig, ax = plt.subplots(figsize=(12, 12))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")

def evaluate_model(model_name):
    print(f"Evaluating {model_name}...")
    experiment_dir = os.path.join("experiments", model_name)
    model_path = os.path.join(experiment_dir, "best_model.h5")
    
    if not os.path.exists(model_path):
        print(f"Skipping {model_name}: No model file found at {model_path}")
        return None

    # Load Data
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    # Load Model
    try:
        model = get_model(model_name, (IMAGE_SIZE, IMAGE_SIZE, 3), test_gen.num_classes)
        model.load_weights(model_path)
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None

    # Predict
    y_true = test_gen.classes
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"{model_name} Accuracy: {acc:.4f}")

    # Confusion Matrix
    class_names = sorted(list(test_gen.class_indices.keys()))
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    
    print(f"CM Shape: {cm.shape}")
    print(f"Class Names Length: {len(class_names)}")
    print(f"Unique y_true: {np.unique(y_true)}")
    print(f"Unique y_pred: {np.unique(y_pred)}")
    
    # Save as CSV
    cm_csv_path = os.path.join(experiment_dir, "confusion_matrix.csv")
    try:
        if cm.shape[0] == len(class_names):
            pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(cm_csv_path)
            print(f"Confusion matrix CSV saved to {cm_csv_path}")
        else:
            print(f"Shape mismatch: CM {cm.shape} vs Classes {len(class_names)}. Saving raw CM.")
            pd.DataFrame(cm).to_csv(cm_csv_path)
    except Exception as e:
        print(f"Error saving CM CSV: {e}")
    
    # Try to plot
    cm_plot_path = os.path.join(experiment_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_plot_path)

    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1,
        "Confusion_Matrix_CSV": cm_csv_path,
        "Confusion_Matrix_Plot": cm_plot_path
    }

def main():
    results = []
    for model_name in MODELS:
        res = evaluate_model(model_name)
        if res:
            results.append(res)
            
    if results:
        df = pd.DataFrame(results)
        df.to_csv("final_evaluation_report.csv", index=False)
        print("\nEvaluation complete. Report saved to final_evaluation_report.csv")
        print(df)
    else:
        print("No models were evaluated.")

if __name__ == "__main__":
    main()
