import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from translate import label_map  

import os
MODEL_PATH = os.path.join(os.path.dirname(__file__), "combined_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

with open("classes.json", "r", encoding="utf-8") as f:
    class_indices = json.load(f)

idx_to_label = {v: k for k, v in class_indices.items()}


model = tf.keras.models.load_model("combined_model.h5")

st.set_page_config(
    page_title="Image Recognition (Hindi/Marathi)",
    layout="centered"
)

st.title("📷 Image Recognition (English → Hindi → Marathi)")

st.write("Upload an image, and the model will classify it and show translations.")

# ---------------------------------------
# IMAGE UPLOADER
# ---------------------------------------
uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    # ---------------------------------------
    # PREPROCESS IMAGE
    # ---------------------------------------
    img_resized = img.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ---------------------------------------
    # MODEL PREDICTION
    # ---------------------------------------
    pred = model.predict(img_array)[0]
    pred_idx = int(np.argmax(pred))
    english_label = idx_to_label[pred_idx]

    st.subheader("Prediction Results")

    # ---------------------------------------
    # TRANSLATIONS
    # ---------------------------------------
    hindi = label_map.get(english_label, {}).get("hi", "—")
    marathi = label_map.get(english_label, {}).get("mr", "—")

    st.write("**English:**", english_label)
    st.write("**Hindi:**", hindi)
    st.write("**Marathi:**", marathi)

    # ---------------------------------------
    # TOP-3 PREDICTIONS
    # ---------------------------------------
    st.write("### Top-3 Predictions:")
    sorted_idx = np.argsort(pred)[::-1][:3]

    for idx in sorted_idx:
        lbl = idx_to_label[idx]
        score = float(pred[idx])
        st.write(f"{lbl} — {score:.3f}")
