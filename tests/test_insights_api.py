import io

import numpy as np
import pytest
from PIL import Image

from app import create_app
from tests.test_app_auth import GREEN, RED, StubVerifier, login, register
from tests.test_replay import make_replay_zip


@pytest.fixture
def make_client(tmp_path):
    def build(replay=True):
        config = {
            "TESTING": True,
            "SECRET_KEY": "test",
            "DB_PATH": str(tmp_path / "test.db"),
            "PHOTO_DIR": str(tmp_path / "photos"),
            "RETRAIN_DIR": str(tmp_path / "retrain"),
            "REPLAY_DIR": str(tmp_path / "replay"),
            "INSIGHT_DIR": str(tmp_path / "insights"),
            "REPLAY_ZIP": str(make_replay_zip(tmp_path / "replay.zip", 12, 4))
            if replay
            else str(tmp_path / "missing.zip"),
            "VERIFIER": StubVerifier(),
        }
        client = create_app(config).test_client()
        register(client, "alice")
        login(client, "alice", RED)
        return client

    return build


def test_insights_require_login(make_client):
    client = make_client()
    client.post("/logout")
    assert client.get("/api/insights").status_code == 302


def test_quality_is_reported_for_each_enrollment_photo(make_client):
    data = make_client().get("/api/insights").get_json()
    assert len(data["quality"]["photos"]) == 3
    assert 0.0 <= data["quality"]["overall"] <= 1.0
    assert all("hints" in p for p in data["quality"]["photos"])


def test_face_map_and_lookalikes_come_from_the_replay_set(make_client):
    data = make_client().get("/api/insights").get_json()
    assert data["map"]["available"] is True
    assert len(data["map"]["dataset"]) > 5 and len(data["map"]["user"]) == 2
    assert 1 <= len(data["lookalikes"]) <= 3
    assert {"name", "similarity", "photo_id"} <= set(data["lookalikes"][0])


def test_other_users_appear_on_the_map_without_their_names(tmp_path, make_client):
    client = make_client()
    other = client.application.test_client()
    register(other, "bobby", color=GREEN)
    data = client.get("/api/insights").get_json()
    assert len(data["map"]["others"]) == 1
    assert "bobby" not in str(data)


def test_without_a_replay_set_the_map_is_unavailable_but_the_rest_works(make_client):
    data = make_client(replay=False).get("/api/insights").get_json()
    assert data["map"]["available"] is False and data["lookalikes"] == []
    assert len(data["quality"]["photos"]) == 3


def test_login_margin_reflects_logins(make_client):
    client = make_client()
    login(client, "alice", GREEN)  # a failed attempt
    margin = client.get("/api/insights").get_json()["margin"]
    assert margin["attempts"] == 2 and margin["successes"] == 1
    assert margin["margin"] is not None


def test_lookalike_photo_endpoint_serves_a_jpeg_and_rejects_bad_ids(make_client):
    client = make_client()
    photo_id = client.get("/api/insights").get_json()["lookalikes"][0]["photo_id"]
    response = client.get(f"/api/insights/photo/{photo_id}")
    assert response.status_code == 200 and response.mimetype == "image/jpeg"
    assert client.get("/api/insights/photo/99999").status_code == 404


def test_users_can_only_fetch_their_own_enrollment_photos(make_client):
    client = make_client()
    assert client.get("/api/insights/my-photo/0").mimetype == "image/jpeg"
    assert client.get("/api/insights/my-photo/9").status_code == 404
