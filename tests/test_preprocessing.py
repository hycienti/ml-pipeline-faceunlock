from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.preprocessing import detect_and_crop

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
