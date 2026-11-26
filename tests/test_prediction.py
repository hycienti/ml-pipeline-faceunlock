from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.prediction import FaceVerifier

LFW_DIR = Path(__file__).parent.parent / "data/lfw/lfw-deepfunneled/lfw-deepfunneled"


class ThumbnailModel:
    """Stand-in for the CNN: an 8x8 grayscale thumbnail, centred and unit length."""

    def predict(self, x, batch_size=None, verbose=0):
        thumb = x[:, ::20, ::20, 0].reshape(len(x), -1)
        thumb = thumb - thumb.mean(axis=1, keepdims=True)
        return thumb / (np.linalg.norm(thumb, axis=1, keepdims=True) + 1e-9)


def load(person, number=1):
    path = LFW_DIR / person / f"{person}_{number:04d}.jpg"
    if not path.exists():
        pytest.skip("LFW not downloaded")
    return np.asarray(Image.open(path).convert("RGB"))


@pytest.fixture
def verifier():
    return FaceVerifier(ThumbnailModel(), threshold=0.9)


def test_embed_returns_none_without_a_face(verifier):
    assert verifier.embed(np.zeros((300, 300, 3), dtype=np.uint8)) is None


def test_embed_returns_unit_vector_for_a_face(verifier):
    embedding = verifier.embed(load("George_W_Bush"))
    assert embedding.shape == (64,)
    assert np.isclose(np.linalg.norm(embedding), 1.0)


def test_enroll_averages_photos_into_a_unit_vector(verifier):
    photos = [load("George_W_Bush", n) for n in (1, 2, 3)]
    template = verifier.enroll(photos)
    assert np.isclose(np.linalg.norm(template), 1.0)


def test_enroll_rejects_a_photo_without_a_face(verifier):
    photos = [load("George_W_Bush"), np.zeros((300, 300, 3), dtype=np.uint8)]
    with pytest.raises(ValueError, match="photo 2"):
        verifier.enroll(photos)


def test_verify_accepts_same_photo_and_rejects_another_person(verifier):
    bush = load("George_W_Bush")
    template = verifier.enroll([bush])
    accepted, score = verifier.verify(bush, template)
    assert accepted and score > 0.99
    accepted, score = verifier.verify(load("Colin_Powell", 2), template)
    assert score > -1.0  # a face WAS found: the rejection comes from the score, not a missing face
    assert not accepted


def test_verify_fails_cleanly_when_no_face(verifier):
    template = verifier.enroll([load("George_W_Bush")])
    accepted, score = verifier.verify(np.zeros((300, 300, 3), dtype=np.uint8), template)
    assert accepted is False and score < 0


def test_embed_many_matches_embed_keeps_order_and_marks_failures(verifier):
    bush, powell = load("George_W_Bush"), load("Colin_Powell", 2)
    blank = np.zeros((300, 300, 3), dtype=np.uint8)
    results = verifier.embed_many([bush, blank, powell])
    assert results[1] is None
    assert np.allclose(results[0], verifier.embed(bush))
    assert np.allclose(results[2], verifier.embed(powell))
    assert verifier.embed_many([]) == []
