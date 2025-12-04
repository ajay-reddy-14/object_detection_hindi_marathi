import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from hybrid_model import HybridEnsemble

# Configuration
DATASET_ROOT = "new_dataset"
TEST_DIR = os.path.join(DATASET_ROOT, "test")
EXPERIMENTS_DIR = "experiments"
IMAGE_SIZE = 224
BATCH_SIZE = 32

def get_best_models():
    """
    Finds the best model checkpoints in the experiments directory.
    Returns a list of paths to .h5 files.
    """
    model_paths = []
    if not os.path.exists(EXPERIMENTS_DIR):
        print("Experiments directory not found.")
        return []

    for model_name in os.listdir(EXPERIMENTS_DIR):
        model_dir = os.path.join(EXPERIMENTS_DIR, model_name)
        if os.path.isdir(model_dir):
            # Prefer best_model.h5, fallback to model.h5
            best_path = os.path.join(model_dir, "best_model.h5")
            final_path = os.path.join(model_dir, "model.h5")
            
            if os.path.exists(best_path):
                model_paths.append(best_path)
            elif os.path.exists(final_path):
                model_paths.append(final_path)
    
    return model_paths

def evaluate_ensemble():
    model_paths = get_best_models()
    if not model_paths:
        print("No trained models found to ensemble.")
        return

    print(f"Found {len(model_paths)} models for ensemble.")
    
    # Initialize Ensemble
    ensemble = HybridEnsemble(model_paths)
    
    # Data Generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False 
    )
    
    y_true = test_gen.classes
    
    # Load weights if available
    weights = None
    if os.path.exists("ensemble_weights.csv"):
        print("Loading optimized weights from ensemble_weights.csv...")
        weights_df = pd.read_csv("ensemble_weights.csv")
        # Ensure weights are in the same order as ensemble.model_names
        weights = []
        for name in ensemble.model_names:
            w = weights_df.loc[weights_df['Model'] == name, 'Weight'].values
            if len(w) > 0:
                weights.append(w[0])
            else:
                print(f"Warning: No weight found for {name}, using default.")
                weights.append(1.0/len(ensemble.model_names))
    
    # Predict
    y_pred_probs = ensemble.predict(test_gen, weights=weights)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    try:
        auc = roc_auc_score(tf.keras.utils.to_categorical(y_true, num_classes=len(np.unique(y_true))), y_pred_probs, multi_class='ovr')
    except Exception as e:
        print(f"Could not calculate AUC: {e}")
        auc = 0.0
        
    metrics = {
        "Model": "Hybrid_Ensemble",
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1,
        "AUC": auc
    }
    
    print("\n" + "="*30)
    print("HYBRID ENSEMBLE RESULTS")
    print("="*30)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    print("="*30)
    
    # Save results
    pd.DataFrame([metrics]).to_csv("hybrid_results.csv", index=False)
    print("Results saved to hybrid_results.csv")

if __name__ == "__main__":
    evaluate_ensemble()
