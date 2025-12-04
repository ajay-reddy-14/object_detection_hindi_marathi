import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model_factory import get_model
import pandas as pd

# Configuration
MODEL_NAME = "mobilenetv2"
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
DATASET_ROOT = "new_dataset"
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VALID_DIR = os.path.join(DATASET_ROOT, "valid")
TEST_DIR = os.path.join(DATASET_ROOT, "test")
EXPERIMENT_DIR = os.path.join("experiments", MODEL_NAME)

os.makedirs(EXPERIMENT_DIR, exist_ok=True)

def train_and_evaluate():
    print(f"=== Training {MODEL_NAME} ===")
    
    # Check available classes
    if not os.path.exists(TRAIN_DIR):
        print(f"Error: Training directory {TRAIN_DIR} not found.")
        return
        
    classes = sorted(os.listdir(TRAIN_DIR))
    print(f"Found {len(classes)} classes: {classes}")
    
    # Simple Data Generators (No complex augmentation)
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    val_gen = test_datagen.flow_from_directory(
        VALID_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    # Model
    model = get_model(MODEL_NAME, (IMAGE_SIZE, IMAGE_SIZE, 3), train_gen.num_classes)

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(EXPERIMENT_DIR, "best_model.h5"), save_best_only=True, monitor='val_accuracy'),
        tf.keras.callbacks.CSVLogger(os.path.join(EXPERIMENT_DIR, "training_log.csv"))
    ]

    # Train
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Evaluate
    print("Evaluating on Test Set...")
    # Load best model
    if os.path.exists(os.path.join(EXPERIMENT_DIR, "best_model.h5")):
        model.load_weights(os.path.join(EXPERIMENT_DIR, "best_model.h5"))
    
    loss, acc = model.evaluate(test_gen)
    print(f"Test Accuracy: {acc:.4f}")

    metrics = {
        "Model": MODEL_NAME,
        "Accuracy": acc
    }
    
    # Save Metrics
    pd.DataFrame([metrics]).to_csv(os.path.join(EXPERIMENT_DIR, "metrics.csv"), index=False)
    
    # Save Classes
    with open(os.path.join(EXPERIMENT_DIR, "classes.json"), "w") as f:
        json.dump(train_gen.class_indices, f)
    
    print(f"Training complete. Model saved to {EXPERIMENT_DIR}")

if __name__ == "__main__":
    train_and_evaluate()
