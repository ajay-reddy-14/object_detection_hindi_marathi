import tensorflow as tf
from tensorflow.keras import layers, models

def get_model(model_name, input_shape, num_classes):
    """
    Returns a compiled Keras model based on the model_name.
    """
    model_name = model_name.lower()
    weights = "imagenet"
    
    model_map = {
        "mobilenetv2": tf.keras.applications.MobileNetV2,
        "resnet50": tf.keras.applications.ResNet50,
        "vgg16": tf.keras.applications.VGG16,
        "efficientnetb0": tf.keras.applications.EfficientNetB0,
        "densenet121": tf.keras.applications.DenseNet121,
        "inceptionv3": tf.keras.applications.InceptionV3,
    }

    if model_name not in model_map:
        raise ValueError(f"Model {model_name} not supported. Choose from: {list(model_map.keys())}")

    # InceptionV3 requires larger input, but we'll let Keras handle the error if shape is too small
    base_model = model_map[model_name](
        input_shape=input_shape, include_top=False, weights=weights
    )

    # Freeze base model
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
