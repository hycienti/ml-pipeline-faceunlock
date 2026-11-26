from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.preprocessing import (
    build_pairs,
    center_crop,
    detect_and_crop,
    preprocess,
    resize_face,
    scale_pixels,
)

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


INDEX = {
    "A": ["a1", "a2", "a3"],
    "B": ["b1", "b2"],
    "C": ["c1"],
    "D": ["d1", "d2"],
}


def owner(path):
    return path[0].upper()


def test_pairs_are_balanced():
    pairs = build_pairs(INDEX, 10, seed=0)
    assert len(pairs) == 10
    assert sum(label for _, _, label in pairs) == 5


def test_labels_match_identity():
    for a, b, label in build_pairs(INDEX, 200, seed=1):
        assert (owner(a) == owner(b)) == (label == 1)


def test_positive_pairs_use_two_different_photos():
    for a, b, label in build_pairs(INDEX, 200, seed=1):
        if label == 1:
            assert a != b


def test_same_seed_is_repeatable_and_different_seed_differs():
    assert build_pairs(INDEX, 20, seed=0) == build_pairs(INDEX, 20, seed=0)
    assert build_pairs(INDEX, 20, seed=0) != build_pairs(INDEX, 20, seed=1)


def test_single_photo_person_never_in_positive_pair():
    for a, b, label in build_pairs(INDEX, 200, seed=2):
        if label == 1:
            assert owner(a) != "C"


def test_impossible_requests_raise():
    with pytest.raises(ValueError):
        build_pairs({"A": ["a1"], "B": ["b1"]}, 4, seed=0)  # nobody has 2 photos
    with pytest.raises(ValueError):
        build_pairs({"A": ["a1", "a2"]}, 4, seed=0)  # only one person


def test_resize_face_returns_uint8_square():
    out = resize_face(np.zeros((120, 80, 3), dtype=np.uint8))
    assert out.shape == (160, 160, 3) and out.dtype == np.uint8


def test_scale_pixels_maps_ends_to_minus_one_and_one():
    assert np.allclose(scale_pixels(np.array([0, 255], dtype=np.uint8)), [-1.0, 1.0])


def test_center_crop_takes_the_middle():
    image = np.arange(100).reshape(10, 10)
    crop = center_crop(image, 0.5)
    assert crop.shape == (5, 5)
    assert np.array_equal(crop, image[2:7, 2:7])
