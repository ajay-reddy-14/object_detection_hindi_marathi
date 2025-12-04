import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from model_factory import get_model

# Parse arguments
parser = argparse.ArgumentParser(description="Train a specific model.")
parser.add_argument("--model", type=str, default="efficientnetb0", help="Model architecture to train")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for initial training")
parser.add_argument("--fine_tune_epochs", type=int, default=10, help="Number of epochs for fine-tuning")
args = parser.parse_args()

MODEL_NAME = args.model
EPOCHS_STAGE1 = args.epochs
EPOCHS_FINE = args.fine_tune_epochs

IMAGE_SIZE = 224
BATCH_SIZE = 32
DATASET_ROOT = "new_dataset"
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VALID_DIR = os.path.join(DATASET_ROOT, "valid")

# Experiment Directory
EXPERIMENT_DIR = os.path.join("experiments", MODEL_NAME)
os.makedirs(EXPERIMENT_DIR, exist_ok=True)
print(f"Training {MODEL_NAME}... Output will be saved to {EXPERIMENT_DIR}")

# Data Generators
train_gen_obj = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True
)
val_gen_obj = ImageDataGenerator(rescale=1./255)

train_gen = train_gen_obj.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_gen = val_gen_obj.flow_from_directory(
    VALID_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

num_classes = train_gen.num_classes
class_indices = train_gen.class_indices

# Class Weights
y_train = train_gen.classes
classes = np.unique(y_train)
cw = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = {int(c): float(w) for c, w in zip(classes, cw)}

# Build Model
model, base_model = get_model(MODEL_NAME, (IMAGE_SIZE, IMAGE_SIZE, 3), num_classes)

# Callbacks
checkpoint_path = os.path.join(EXPERIMENT_DIR, "best_model.h5")
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max'),
    tf.keras.callbacks.CSVLogger(os.path.join(EXPERIMENT_DIR, "training_log.csv"))
]

# Stage 1 Training
print("Starting Stage 1 Training...")
history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_STAGE1,
    class_weight=class_weights,
    callbacks=callbacks
)

# Fine-tuning
print("Starting Fine-tuning...")
base_model.trainable = True
# Freeze earlier layers
# Note: This is a simple heuristic. For deeper models, you might want to freeze more/less.
# For simplicity, we freeze the first 70% of layers.
num_layers = len(base_model.layers)
freeze_until = int(num_layers * 0.7)
for layer in base_model.layers[:freeze_until]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINE,
    class_weight=class_weights,
    callbacks=callbacks
)

# Save Final Model
final_model_path = os.path.join(EXPERIMENT_DIR, "model.h5")
model.save(final_model_path)

# Save Classes
with open(os.path.join(EXPERIMENT_DIR, "classes.json"), "w", encoding="utf-8") as f:
    json.dump(class_indices, f, indent=2, ensure_ascii=False)

print(f"Training complete for {MODEL_NAME}. Model saved to {final_model_path}")
