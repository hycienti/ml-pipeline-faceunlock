import numpy as np
import pytest
from tensorflow import keras

from src.model import (
    CosineMarginHead,
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
    onehot = np.zeros((4, 830), dtype="float32")
    assert model.predict([batch, onehot], verbose=0).shape == (4, 830)


def test_l2_zero_turns_weight_decay_off():
    assert len(build_embedding_model(l2=0.0).losses) == 0
    assert len(build_embedding_model(l2=1e-4).losses) == 1


def test_training_model_can_skip_augmentation():
    with_aug = build_training_model(build_embedding_model(), 10, augment=True)
    without = build_training_model(build_embedding_model(), 10, augment=False)
    assert any(layer.name == "augmentation" for layer in with_aug.layers)
    assert not any(layer.name == "augmentation" for layer in without.layers)


def test_cosine_head_subtracts_scaled_margin_from_the_true_class_only():
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(5, 16)).astype("float32")
    onehot = np.eye(4, dtype="float32")[[0, 1, 2, 3, 0]]
    head = CosineMarginHead(4, scale=10.0, margin=0.3)
    with_margin = np.asarray(head([embeddings, onehot]))
    without = np.asarray(head([embeddings, np.zeros_like(onehot)]))
    assert np.allclose(without - with_margin, 10.0 * 0.3 * onehot, atol=1e-4)
    assert np.abs(without).max() <= 10.0 + 1e-4  # cosine similarity is bounded by 1
