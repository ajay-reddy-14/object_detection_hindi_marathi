import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = 128

# Load model safely
MODEL_PATH = os.path.join(os.path.dirname(__file__), "combined_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

with open("classes.json", "r") as f:
    class_indices = json.load(f)

idx_to_label = {v:k for k,v in class_indices.items()}

DATA = "new_dataset/train"

print("\n==== TESTING ONE IMAGE PER CLASS ====\n")

for cls in os.listdir(DATA):
    cls_path = os.path.join(DATA, cls)
    if not os.path.isdir(cls_path):
        continue
    
    img_list = os.listdir(cls_path)
    if len(img_list) == 0:
        continue
    
    img_path = os.path.join(cls_path, img_list[0])
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    x = np.array(img)/255.0
    x = np.expand_dims(x, 0)

    pred = model.predict(x)[0]
    pred_idx = np.argmax(pred)
    pred_label = idx_to_label[pred_idx]
    
    print(f"Actual: {cls} → Predicted: {pred_label}")
