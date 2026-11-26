from tensorflow import keras
from tensorflow.keras import layers, regularizers

INPUT_SIZE = 160


def build_augmentation() -> keras.Sequential:
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(10 / 360),
            layers.RandomZoom(0.1),
            layers.RandomBrightness(0.2, value_range=(-1, 1)),
            layers.RandomContrast(0.2),
        ],
        name="augmentation",
    )


def build_embedding_model(
    embedding_dim: int = 128,
    trainable_backbone: bool = False,
    dropout: float = 0.3,
    l2: float = 1e-4,
) -> keras.Model:
    backbone = keras.applications.MobileNetV2(
        input_shape=(INPUT_SIZE, INPUT_SIZE, 3), include_top=False, weights="imagenet"
    )
    backbone.trainable = trainable_backbone

    inputs = keras.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(
        embedding_dim, kernel_regularizer=regularizers.l2(l2) if l2 else None
    )(x)
    outputs = layers.UnitNormalization()(x)
    return keras.Model(inputs, outputs, name="embedding")


def unfreeze_top_layers(model: keras.Model, n: int = 30) -> None:
    backbone = next(layer for layer in model.layers if isinstance(layer, keras.Model))
    backbone.trainable = True
    for layer in backbone.layers[:-n]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def build_training_model(
    embedding_model: keras.Model, n_classes: int, augment: bool = True
) -> keras.Model:
    inputs = keras.Input(shape=embedding_model.input_shape[1:])
    x = build_augmentation()(inputs) if augment else inputs
    embedding = embedding_model(x)
    logits = layers.Dense(n_classes, name="classifier")(embedding)
    return keras.Model(inputs, logits, name="training_model")
