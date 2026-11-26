from dataclasses import dataclass

import tensorflow as tf
from tensorflow import keras

from src.evaluation import evaluate_pairs, pair_scores
from src.model import (
    build_embedding_model,
    build_training_model,
    freeze_backbone,
    unfreeze_top_layers,
)
from src.preprocessing import scale_pixels


@dataclass(frozen=True)
class RunConfig:
    name: str
    augment: bool = False
    dropout: float = 0.0
    l2: float = 0.0
    label_smoothing: float = 0.0
    early_stopping: bool = False
    fine_tune: bool = False
    unfreeze_layers: int = 30
    learning_rates: tuple = (1e-3, 1e-5)


def make_dataset(faces, labels, n_classes, batch_size=32, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((faces, labels))
    if shuffle:
        ds = ds.shuffle(len(faces), seed=0)
    ds = ds.batch(batch_size)
    ds = ds.map(
        lambda x, y: (tf.cast(x, tf.float32) / 127.5 - 1.0, tf.one_hot(y, n_classes))
    )
    return ds.prefetch(tf.data.AUTOTUNE)


class PairAUC(keras.callbacks.Callback):
    def __init__(self, embedding_model, faces, rows_a, rows_b, labels):
        super().__init__()
        self.embedding_model = embedding_model
        self.x = scale_pixels(faces)
        self.rows_a, self.rows_b, self.labels = rows_a, rows_b, labels

    def on_epoch_end(self, epoch, logs=None):
        embeddings = self.embedding_model.predict(self.x, batch_size=64, verbose=0)
        scores = pair_scores(embeddings, self.rows_a, self.rows_b)
        logs["val_auc"] = evaluate_pairs(scores, self.labels)["auc"]


def train(
    config, train_data, val_data, n_classes, epochs=30, batch_size=32, seed=0,
    embedding=None, extra_callbacks=(),
):
    keras.utils.set_random_seed(seed)
    x_train, y_train = train_data
    train_ds = make_dataset(x_train, y_train, n_classes, batch_size, shuffle=True)

    if embedding is None:
        embedding = build_embedding_model(dropout=config.dropout, l2=config.l2)
    else:
        freeze_backbone(embedding)
    model = build_training_model(embedding, n_classes, augment=config.augment)
    pair_auc = PairAUC(embedding, *val_data)

    first_lr, second_lr = config.learning_rates
    phases = [(first_lr, False)] + ([(second_lr, True)] if config.fine_tune else [])
    history = {"loss": [], "val_auc": []}
    for learning_rate, fine_tune in phases:
        if fine_tune:
            unfreeze_top_layers(embedding, config.unfreeze_layers)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate, clipnorm=1.0),
            loss=keras.losses.CategoricalCrossentropy(
                from_logits=True, label_smoothing=config.label_smoothing
            ),
        )
        callbacks = [
            pair_auc,
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_auc", mode="max", factor=0.5, patience=2
            ),
        ]
        callbacks += list(extra_callbacks)
        if config.early_stopping:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_auc", mode="max", patience=5, restore_best_weights=True
                )
            )
        fit = model.fit(train_ds, epochs=epochs, callbacks=callbacks, verbose=0)
        history["loss"] += fit.history["loss"]
        history["val_auc"] += fit.history["val_auc"]
    return embedding, history
