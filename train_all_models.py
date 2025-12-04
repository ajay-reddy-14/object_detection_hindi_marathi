import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model_factory import get_model
import pandas as pd
import time

# Configuration
MODELS_TO_TRAIN = ["mobilenetv2", "resnet50", "vgg16", "efficientnetb0", "densenet121", "inceptionv3"]
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
DATASET_ROOT = "new_dataset"
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VALID_DIR = os.path.join(DATASET_ROOT, "valid")
TEST_DIR = os.path.join(DATASET_ROOT, "test")

def train_single_model(model_name):
    print(f"\n{'='*40}")
    print(f"STARTING TRAINING: {model_name}")
    print(f"{'='*40}\n")
    
    experiment_dir = os.path.join("experiments", model_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Data Generators
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
    try:
        model = get_model(model_name, (IMAGE_SIZE, IMAGE_SIZE, 3), train_gen.num_classes)
    except Exception as e:
        print(f"Error creating model {model_name}: {e}")
        return None

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(experiment_dir, "best_model.h5"), save_best_only=True, monitor='val_accuracy'),
        tf.keras.callbacks.CSVLogger(os.path.join(experiment_dir, "training_log.csv"))
    ]

    # Train
    start_time = time.time()
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    duration = time.time() - start_time

    # Evaluate
    print(f"Evaluating {model_name}...")
    if os.path.exists(os.path.join(experiment_dir, "best_model.h5")):
        model.load_weights(os.path.join(experiment_dir, "best_model.h5"))
    
    loss, acc = model.evaluate(test_gen)
    print(f"{model_name} Test Accuracy: {acc:.4f}")

    metrics = {
        "Model": model_name,
        "Accuracy": acc,
        "Training_Time_Sec": duration
    }
    
    # Save Metrics
    pd.DataFrame([metrics]).to_csv(os.path.join(experiment_dir, "metrics.csv"), index=False)
    
    # Save Classes
    with open(os.path.join(experiment_dir, "classes.json"), "w") as f:
        json.dump(train_gen.class_indices, f)
        
    return metrics

def main():
    if not os.path.exists(TRAIN_DIR):
        print("Error: Dataset not found.")
        return

    all_results = []
    
    for model_name in MODELS_TO_TRAIN:
        result = train_single_model(model_name)
        if result:
            all_results.append(result)
            
    # Save Summary
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv("model_comparison_results.csv", index=False)
        print("\nAll training complete. Results saved to model_comparison_results.csv")
        print(df)

if __name__ == "__main__":
    main()
