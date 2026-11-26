import numpy as np
import pytest

from src.live_evaluation import evaluate_live
from src.replay import load_replay
from tests.test_replay import make_replay_zip


class TwoClassVerifier:
    """Embeds even-numbered images as [1, 0] and odd ones as [0, 1]; optionally finds no face."""

    threshold = 0.5

    def __init__(self, no_face=()):
        self.no_face = set(no_face)

    def embed_many(self, images):
        return [
            None if i in self.no_face else (np.array([1.0, 0.0]) if i % 2 == 0 else np.array([0.0, 1.0]))
            for i in range(len(images))
        ]


@pytest.fixture
def replay(tmp_path):
    return load_replay(str(make_replay_zip(tmp_path / "r.zip", 6, 6)), str(tmp_path / "out"))


def test_live_evaluation_reports_the_course_metrics_at_the_live_threshold(replay):
    result = evaluate_live(TwoClassVerifier(), replay)
    assert {"accuracy", "precision", "recall", "f1", "auc", "far", "frr"} <= set(result["metrics"])
    assert result["threshold"] == 0.5 and result["seconds"] >= 0
    assert result["pairs_evaluated"] + result["pairs_skipped_no_face"] == len(replay["val_pairs"])
    assert "fpr" not in result["metrics"]  # curves are not stored per run


def test_pairs_the_live_pipeline_cannot_read_are_skipped_and_counted(replay):
    everything = evaluate_live(TwoClassVerifier(), replay)
    some_missing = evaluate_live(TwoClassVerifier(no_face=range(0, 8)), replay)
    assert everything["pairs_skipped_no_face"] == 0
    assert some_missing["pairs_skipped_no_face"] > 0
    assert some_missing["pairs_evaluated"] < everything["pairs_evaluated"]


def test_nothing_readable_raises_a_clear_error(replay):
    with pytest.raises(ValueError, match="no usable"):
        evaluate_live(TwoClassVerifier(no_face=range(1000)), replay)
