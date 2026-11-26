from pathlib import Path

import numpy as np
import pytest
from tensorflow import keras

MODELS = Path(__file__).parent.parent / "models"


def test_shipped_h5_and_keras_models_give_the_same_embeddings():
    images = np.random.default_rng(0).uniform(-1, 1, (4, 160, 160, 3)).astype("float32")
    native = keras.models.load_model(MODELS / "faceunlock_v1.keras")
    legacy = keras.models.load_model(MODELS / "faceunlock_v1.h5")
    expected, got = native.predict(images, verbose=0), legacy.predict(images, verbose=0)
    assert got.shape == (4, 128)
    assert np.allclose(expected, got, atol=1e-4)
    assert np.allclose(np.linalg.norm(got, axis=1), 1.0, atol=1e-4)
