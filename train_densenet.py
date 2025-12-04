import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from model_factory import get_model
import pandas as pd

# Configuration
MODEL_NAME = "densenet121"
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
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
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

    y_train = train_gen.classes
    classes = np.unique(y_train)
    cw = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(classes, cw)}

    model, base_model = get_model(MODEL_NAME, (IMAGE_SIZE, IMAGE_SIZE, 3), train_gen.num_classes)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(EXPERIMENT_DIR, "best_model.h5"), save_best_only=True, monitor='val_accuracy'),
        tf.keras.callbacks.CSVLogger(os.path.join(EXPERIMENT_DIR, "training_log.csv"))
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks
    )

    print("Evaluating on Test Set...")
    model.load_weights(os.path.join(EXPERIMENT_DIR, "best_model.h5"))
    
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
        "Model": MODEL_NAME,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1,
        "AUC": auc
    }
    
    print("Metrics:", metrics)
    pd.DataFrame([metrics]).to_csv(os.path.join(EXPERIMENT_DIR, "metrics.csv"), index=False)
    
    with open(os.path.join(EXPERIMENT_DIR, "classes.json"), "w") as f:
        json.dump(train_gen.class_indices, f)

if __name__ == "__main__":
    train_and_evaluate()
