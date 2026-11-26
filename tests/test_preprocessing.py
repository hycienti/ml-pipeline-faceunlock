from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.preprocessing import detect_and_crop, preprocess

LFW_DIR = Path(__file__).parent.parent / "data/lfw/lfw-deepfunneled/lfw-deepfunneled"


@pytest.fixture
def face():
    path = LFW_DIR / "George_W_Bush" / "George_W_Bush_0001.jpg"
    if not path.exists():
        pytest.skip("LFW not downloaded")
    return np.asarray(Image.open(path).convert("RGB"))


def test_blank_image_has_no_face():
    assert detect_and_crop(np.zeros((300, 300, 3), dtype=np.uint8)) is None


def test_single_face_returns_rgb_crop(face):
    crop = detect_and_crop(face)
    assert crop is not None
    assert crop.ndim == 3 and crop.shape[2] == 3


def test_two_faces_are_rejected(face):
    two_faces = np.hstack([face, face])
    assert detect_and_crop(two_faces) is None


def test_preprocess_output_shape_dtype_and_range():
    face = np.random.default_rng(0).integers(0, 256, (200, 300, 3), dtype=np.uint8)
    out = preprocess(face)
    assert out.shape == (160, 160, 3)
    assert out.dtype == np.float32
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_preprocess_accepts_grayscale_and_rgba():
    gray = np.full((100, 100), 128, dtype=np.uint8)
    rgba = np.full((100, 100, 4), 128, dtype=np.uint8)
    assert preprocess(gray).shape == (160, 160, 3)
    assert preprocess(rgba).shape == (160, 160, 3)


def test_preprocess_maps_black_and_white_to_range_ends():
    black = np.zeros((50, 50, 3), dtype=np.uint8)
    white = np.full((50, 50, 3), 255, dtype=np.uint8)
    assert np.allclose(preprocess(black), -1.0)
    assert np.allclose(preprocess(white), 1.0)
