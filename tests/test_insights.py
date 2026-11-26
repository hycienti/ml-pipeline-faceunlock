from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from src.insights import (
    dataset_stories,
    embedding_map,
    login_margin,
    nearest_lookalikes,
    quality_score,
)

LFW_DIR = Path(__file__).parent.parent / "data/lfw/lfw-deepfunneled/lfw-deepfunneled"


def noise(seed=0, size=300):
    return np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8)


def test_dark_photo_gets_a_light_hint():
    result = quality_score(np.full((300, 300, 3), 20, dtype=np.uint8))
    assert result["brightness"] < 70
    assert any("dark" in hint.lower() for hint in result["hints"])


def test_blurry_photo_scores_lower_on_sharpness_and_gets_a_hint():
    sharp = noise()
    blurry = cv2.GaussianBlur(sharp, (41, 41), 0)
    assert quality_score(sharp)["sharpness"] > quality_score(blurry)["sharpness"]
    assert any("blurry" in hint.lower() for hint in quality_score(blurry)["hints"])


def test_photo_without_a_face_is_flagged_but_still_scored():
    result = quality_score(np.full((300, 300, 3), 128, dtype=np.uint8))
    assert result["face_ratio"] is None
    assert any("face" in hint.lower() for hint in result["hints"])
    assert 0.0 <= result["overall"] <= 1.0


def test_a_good_real_face_photo_scores_well():
    path = LFW_DIR / "George_W_Bush" / "George_W_Bush_0001.jpg"
    if not path.exists():
        pytest.skip("LFW not downloaded")
    result = quality_score(np.asarray(Image.open(path).convert("RGB")))
    assert 0.2 < result["face_ratio"] < 0.8
    assert result["overall"] > 0.4


def test_embedding_map_projects_everything_to_two_dimensions():
    rng = np.random.default_rng(0)
    dataset = rng.normal(size=(50, 16))
    user = rng.normal(size=16)
    others = rng.normal(size=(3, 16))
    result = embedding_map(dataset, user, others)
    assert len(result["dataset"]) == 50 and len(result["others"]) == 3
    assert len(result["user"]) == 2 and all(len(p) == 2 for p in result["dataset"])
    assert 0.0 < sum(result["explained"]) <= 1.0


def test_embedding_map_keeps_a_far_away_user_far_away():
    rng = np.random.default_rng(1)
    dataset = rng.normal(size=(60, 8))
    far_user = np.full(8, 10.0)
    result = embedding_map(dataset, far_user)
    spread = np.abs(np.array(result["dataset"])).max()
    assert np.abs(result["user"]).max() > spread


def test_nearest_lookalikes_are_sorted_by_similarity():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.6], [-1.0, 0.0]])
    names = ["a", "b", "c", "d"]
    result = nearest_lookalikes(np.array([1.0, 0.0]), embeddings, names, k=3)
    assert [r["name"] for r in result] == ["a", "c", "b"]
    assert result[0]["similarity"] == pytest.approx(1.0)
    assert result[0]["similarity"] > result[1]["similarity"] > result[2]["similarity"]


def test_login_margin_summarises_attempts_against_the_threshold():
    rows = [(1, 0.60), (1, 0.50), (0, 0.10), (0, None)]
    result = login_margin(rows, threshold=0.4)
    assert result["attempts"] == 4 and result["successes"] == 2
    assert result["mean_success_score"] == pytest.approx(0.55)
    assert result["margin"] == pytest.approx(0.15)
    assert len(result["recent"]) == 4 and result["threshold"] == 0.4


def test_login_margin_without_successes_has_no_margin():
    result = login_margin([(0, 0.1)], threshold=0.4)
    assert result["margin"] is None and result["mean_success_score"] is None


STATS = {
    "photo_cap": 20,
    "detector_success_rate": 0.917,
    "lfw": {"people": 5749, "single_photo_people": 4069, "usable_people": 1680,
            "top_person": "George_W_Bush", "top_count": 530},
    "kaggle": {
        "total_images": 7219, "sampled": 1000,
        "colour_modes": {"RGB": 969, "L": 18, "RGBA": 12, "P": 1},
        "width_px": {"min": 236, "median": 633, "max": 7680},
        "brightness_stats": {"mean": 120.3, "std": 38.9, "min": 31.0, "max": 232.6},
    },
}


def test_dataset_stories_cover_at_least_three_features_with_real_numbers():
    stories = dataset_stories(STATS)
    assert len(stories) >= 3 and all({"feature", "story"} <= set(s) for s in stories)
    text = " ".join(s["story"] for s in stories)
    for fact in ["4,069", "70.8%", "1,680", "George W Bush", "530", "7,680", "633", "120", "91.7%"]:
        assert fact in text, fact


def test_dataset_stories_explain_why_each_finding_matters():
    stories = {s["feature"]: s["story"] for s in dataset_stories(STATS)}
    assert "cap" in stories["Photos per person"].lower()
    assert "160" in stories["Image size"]
    assert "augment" in stories["Lighting"].lower()


def test_dataset_stories_tolerate_missing_sections():
    assert dataset_stories({}) == []
    assert len(dataset_stories({"detector_success_rate": 0.9})) == 1
