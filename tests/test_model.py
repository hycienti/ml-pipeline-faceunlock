import numpy as np
import pytest
from tensorflow import keras

from src.model import (
    build_augmentation,
    build_embedding_model,
    build_training_model,
    unfreeze_top_layers,
)


@pytest.fixture(scope="module")
def batch():
    return np.random.default_rng(0).uniform(-1, 1, (4, 160, 160, 3)).astype("float32")


def backbone_of(model):
    return next(layer for layer in model.layers if isinstance(layer, keras.Model))


def test_embedding_has_requested_size_and_unit_norm(batch):
    out = build_embedding_model(embedding_dim=128).predict(batch, verbose=0)
    assert out.shape == (4, 128)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-4)


def test_backbone_is_frozen_by_default():
    model = build_embedding_model()
    assert not backbone_of(model).trainable
    assert len(model.trainable_weights) > 0  # the new head still learns


def test_augmentation_only_changes_images_when_training(batch):
    augment = build_augmentation()
    assert np.allclose(augment(batch, training=False), batch)
    assert not np.allclose(augment(batch, training=True), batch)


def test_unfreeze_top_layers_leaves_batchnorm_frozen():
    model = build_embedding_model()
    unfreeze_top_layers(model, n=30)
    layers_ = backbone_of(model).layers
    assert any(layer.trainable for layer in layers_)
    assert not any(
        layer.trainable
        for layer in layers_
        if isinstance(layer, keras.layers.BatchNormalization)
    )


def test_training_model_outputs_one_score_per_identity(batch):
    model = build_training_model(build_embedding_model(), n_classes=830)
    assert model.predict(batch, verbose=0).shape == (4, 830)
