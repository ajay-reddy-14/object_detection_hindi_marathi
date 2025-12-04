# train.py
import os, json, math
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight

IMAGE_SIZE = 128
BATCH_SIZE = 32
DATASET_ROOT = "new_dataset"   # result of conversion script
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VALID_DIR = os.path.join(DATASET_ROOT, "valid")
EPOCHS_STAGE1 = 8
EPOCHS_FINE = 5

# Data generators
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
print("Detected classes:", num_classes)
print(train_gen.class_indices)

# compute class weights to handle imbalance
y_train = train_gen.classes
classes = np.unique(y_train)
cw = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = {int(c): float(w) for c, w in zip(classes, cw)}
print("Computed class weights:", class_weights)

# Build MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Stage 1 training
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_STAGE1,
    class_weight=class_weights
)

# Fine-tuning: unfreeze last layers
base_model.trainable = True
# Freeze earlier layers to avoid overfitting
for layer in base_model.layers[:-60]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINE,
    class_weight=class_weights
)

# Save model and classes
model.save("combined_model.h5")
with open("classes.json", "w", encoding="utf-8") as f:
    json.dump(train_gen.class_indices, f, indent=2, ensure_ascii=False)

print("Training done. Model saved as combined_model.h5 and classes.json created.")

from sklearn.metrics import classification_report
import numpy as np

val_gen.reset()
preds = model.predict(val_gen)
y_pred = np.argmax(preds, axis=1)
y_true = val_gen.classes
labels = list(val_gen.class_indices.keys())

print("\n==== CLASSIFICATION REPORT ====\n")
print(classification_report(y_true, y_pred, target_names=labels))
