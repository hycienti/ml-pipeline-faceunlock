import tensorflow as tf
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


def freeze_backbone(model: keras.Model) -> None:
    next(layer for layer in model.layers if isinstance(layer, keras.Model)).trainable = False


def unfreeze_top_layers(model: keras.Model, n: int = 30) -> None:
    backbone = next(layer for layer in model.layers if isinstance(layer, keras.Model))
    backbone.trainable = True
    for layer in backbone.layers[:-n]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


class CosineMarginHead(layers.Layer):
    """CosFace-style classifier: scale * (cosine - margin on the true class).

    The margin forces same-person embeddings to be clearly closer than the nearest
    other person, which is exactly what verification needs. Only used while training.
    """

    def __init__(self, n_classes: int, scale: float = 30.0, margin: float = 0.25, **kwargs):
        super().__init__(**kwargs)
        self.n_classes, self.scale, self.margin = n_classes, scale, margin

    def build(self, input_shape):
        self.weights_matrix = self.add_weight(
            shape=(input_shape[0][-1], self.n_classes), initializer="glorot_uniform", name="w"
        )

    def call(self, inputs):
        embeddings, onehot = inputs
        cosine = tf.matmul(
            tf.math.l2_normalize(embeddings, axis=1),
            tf.math.l2_normalize(self.weights_matrix, axis=0),
        )
        return self.scale * (cosine - self.margin * onehot)


def build_training_model(
    embedding_model: keras.Model,
    n_classes: int,
    augment: bool = True,
    scale: float = 30.0,
    margin: float = 0.25,
) -> keras.Model:
    images = keras.Input(shape=embedding_model.input_shape[1:])
    labels = keras.Input(shape=(n_classes,), name="label")
    x = build_augmentation()(images) if augment else images
    logits = CosineMarginHead(n_classes, scale, margin, name="head")(
        [embedding_model(x), labels]
    )
    return keras.Model([images, labels], logits, name="training_model")
