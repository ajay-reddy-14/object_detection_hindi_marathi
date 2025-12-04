import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_PATH = "merged_dataset"
IMG_SIZE = 128
BATCH_SIZE = 32

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

with open("classes.json", "w", encoding="utf-8") as f:
    json.dump(train_gen.class_indices, f, indent=2, ensure_ascii=False)

print("Correct classes.json saved:")
print(train_gen.class_indices)
