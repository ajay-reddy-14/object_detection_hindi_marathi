import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.optimize import minimize
from hybrid_model import HybridEnsemble

# Configuration
DATASET_ROOT = "new_dataset"
VALID_DIR = os.path.join(DATASET_ROOT, "valid") # Optimize on Validation set!
EXPERIMENTS_DIR = "experiments"
IMAGE_SIZE = 224
BATCH_SIZE = 32

def get_best_models():
    model_paths = []
    if not os.path.exists(EXPERIMENTS_DIR):
        return []

    for model_name in os.listdir(EXPERIMENTS_DIR):
        model_dir = os.path.join(EXPERIMENTS_DIR, model_name)
        if os.path.isdir(model_dir):
            best_path = os.path.join(model_dir, "best_model.h5")
            final_path = os.path.join(model_dir, "model.h5")
            if os.path.exists(best_path):
                model_paths.append(best_path)
            elif os.path.exists(final_path):
                model_paths.append(final_path)
    return model_paths

def optimize_weights():
    model_paths = get_best_models()
    if not model_paths:
        print("No models found.")
        return

    ensemble = HybridEnsemble(model_paths)
    
    # Load Validation Data
    val_datagen = ImageDataGenerator(rescale=1./255)
    val_gen = val_datagen.flow_from_directory(
        VALID_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    y_true = val_gen.classes
    y_true_onehot = tf.keras.utils.to_categorical(y_true)
    
    # Pre-calculate predictions for all models
    all_preds = ensemble.get_all_predictions(val_gen)
    
    def loss_func(weights):
        # Normalize weights
        weights = np.array(weights)
        weights = np.maximum(weights, 0) # Ensure non-negative
        if np.sum(weights) == 0:
            return 1000.0 # High loss if all weights 0
        weights = weights / np.sum(weights)
        
        # Weighted average prediction
        weighted_pred = np.average(all_preds, axis=0, weights=weights)
        
        # Clip to avoid log(0)
        weighted_pred = np.clip(weighted_pred, 1e-15, 1 - 1e-15)
        
        # Log Loss (Categorical Crossentropy)
        loss = -np.mean(np.sum(y_true_onehot * np.log(weighted_pred), axis=1))
        return loss

    # Initial weights (equal)
    init_weights = [1.0 / len(ensemble.model_paths)] * len(ensemble.model_paths)
    bounds = [(0, 1)] * len(ensemble.model_paths)
    constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    
    print("Optimizing weights...")
    result = minimize(loss_func, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    best_weights = result.x
    best_weights = best_weights / np.sum(best_weights) # Ensure sum to 1
    
    print("\nOptimized Weights:")
    for name, w in zip(ensemble.model_names, best_weights):
        print(f"{name}: {w:.4f}")
        
    # Save weights
    weights_df = pd.DataFrame({'Model': ensemble.model_names, 'Weight': best_weights})
    weights_df.to_csv("ensemble_weights.csv", index=False)
    print("Weights saved to ensemble_weights.csv")

if __name__ == "__main__":
    optimize_weights()
