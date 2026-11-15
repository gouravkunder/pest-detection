"""ResNet152V2 transfer-learning model construction."""


def build_model(number_of_classes: int, fine_tune_layers: int = 30):
    if number_of_classes < 2:
        raise ValueError("At least two classes are required for classification.")
    if fine_tune_layers < 0:
        raise ValueError("fine_tune_layers cannot be negative.")
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is required to build the model. Install the project "
            "requirements in a Python environment supported by TensorFlow."
        ) from error

    inputs = tf.keras.Input(shape=(224, 224, 3), name="image")
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="training_augmentation",
    )
    images = augmentation(inputs)
    images = tf.keras.layers.Rescaling(
        scale=1.0 / 127.5, offset=-1, name="resnet_v2_preprocessing"
    )(images)
    base_model = tf.keras.applications.ResNet152V2(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
    )
    base_model.trainable = False
    features = base_model(images, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(
        features
    )
    features = tf.keras.layers.BatchNormalization(name="head_batch_normalization")(
        features
    )
    features = tf.keras.layers.Dropout(0.4, name="head_dropout")(features)
    features = tf.keras.layers.Dense(256, activation="relu", name="dense_256")(
        features
    )
    features = tf.keras.layers.Dropout(0.3, name="classifier_dropout")(features)
    outputs = tf.keras.layers.Dense(
        number_of_classes, activation="softmax", name="class_probabilities"
    )(features)
    model = tf.keras.Model(inputs, outputs, name="pest_resnet152v2")
    return model, base_model, augmentation


def configure_fine_tuning(base_model, fine_tune_layers: int) -> int:
    """Unfreeze only the final base layers, leaving all BN layers frozen."""
    import tensorflow as tf

    if fine_tune_layers < 1:
        raise ValueError("fine_tune_layers must be at least 1 for fine-tuning.")
    base_model.trainable = True
    cutoff = max(0, len(base_model.layers) - fine_tune_layers)
    for index, layer in enumerate(base_model.layers):
        layer.trainable = index >= cutoff and not isinstance(
            layer, tf.keras.layers.BatchNormalization
        )
    return sum(layer.trainable for layer in base_model.layers)
