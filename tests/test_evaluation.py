import numpy as np
import pytest

from src.evaluation import evaluate_pairs, pair_scores


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
