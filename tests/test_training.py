import numpy as np
import pytest

from src.preprocessing import scale_pixels
from src.training import RunConfig, make_dataset, train


@pytest.fixture(scope="module")
def tiny():
    rng = np.random.default_rng(0)
    faces = rng.integers(0, 256, (24, 160, 160, 3), dtype=np.uint8)
    labels = np.repeat(np.arange(6), 4)
    val_faces = rng.integers(0, 256, (8, 160, 160, 3), dtype=np.uint8)
    val = (
        val_faces,
        np.array([0, 2, 4, 6]),
        np.array([1, 3, 5, 7]),
        np.array([1, 0, 1, 0]),
    )
    return (faces, labels), val


def test_dataset_scales_pixels_and_one_hot_encodes_labels(tiny):
    (faces, labels), _ = tiny
    (x, y_in), y = next(iter(make_dataset(faces, labels, n_classes=6, batch_size=4)))
    assert np.allclose(x.numpy(), scale_pixels(faces[:4]))
    assert y.shape == (4, 6) and y.numpy().sum() == 4
    assert np.array_equal(y_in.numpy(), y.numpy())  # labels are also model inputs (margin head)


def test_train_returns_unit_norm_embedding_and_history(tiny):
    train_data, val = tiny
    config = RunConfig(name="t", augment=True, dropout=0.3, l2=1e-4, label_smoothing=0.1)
    embedding, history = train(config, train_data, val, n_classes=6, epochs=1, batch_size=8)
    out = embedding.predict(scale_pixels(val[0][:3]), verbose=0)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-4)
    assert len(history["loss"]) == 1
    assert 0.0 <= history["val_auc"][0] <= 1.0


def test_fine_tune_adds_a_second_phase(tiny):
    train_data, val = tiny
    config = RunConfig(name="t", fine_tune=True, early_stopping=True)
    _, history = train(config, train_data, val, n_classes=6, epochs=1, batch_size=8)
    assert len(history["loss"]) == 2
