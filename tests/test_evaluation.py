import numpy as np
import pytest

from src.evaluation import embed_faces, evaluate_pairs, pair_scores, threshold_for_far
from src.preprocessing import scale_pixels


def test_pair_scores_are_cosine_similarity_of_unit_vectors():
    embeddings = np.eye(3)
    scores = pair_scores(embeddings, np.array([0, 0]), np.array([0, 1]))
    assert np.allclose(scores, [1.0, 0.0])


def test_perfectly_separated_pairs():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    result = evaluate_pairs(scores, labels)
    assert result["auc"] == 1.0
    assert result["accuracy"] == 1.0
    assert 0.2 < result["threshold"] <= 0.8


def test_overlapping_pairs_at_fixed_threshold():
    scores = np.array([0.9, 0.4, 0.6, 0.1])
    labels = np.array([1, 1, 0, 0])
    result = evaluate_pairs(scores, labels, threshold=0.5)
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)
    assert result["far"] == pytest.approx(0.5)
    assert result["frr"] == pytest.approx(0.5)
    assert result["auc"] == pytest.approx(0.75)


class FakeModel:
    def predict(self, x, verbose=0):
        return x.reshape(len(x), -1)[:, :2]


def test_embed_faces_batches_and_keeps_order():
    faces = np.arange(5 * 2 * 2 * 3, dtype=np.uint8).reshape(5, 2, 2, 3)
    out = embed_faces(FakeModel(), faces, batch_size=2)
    assert out.shape == (5, 2)
    assert np.allclose(out, scale_pixels(faces).reshape(5, -1)[:, :2])


def test_threshold_for_far_picks_the_loosest_threshold_within_the_stranger_budget():
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.65, 0.3, 0.2, 0.1])
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    assert threshold_for_far(scores, labels, far=0.0) == pytest.approx(0.7)
    assert threshold_for_far(scores, labels, far=0.25) == pytest.approx(0.6)


def test_threshold_for_far_never_returns_infinity():
    scores, labels = np.array([0.5, 0.9]), np.array([0, 1])
    assert np.isfinite(threshold_for_far(scores, labels, far=0.0))
