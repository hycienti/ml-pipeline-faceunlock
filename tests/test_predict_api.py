import io

import pytest
from PIL import Image

from app import create_app
from src.prediction import predict_bar
from tests.test_app_auth import BLACK, BLUE, GREEN, RED, StubVerifier, login, register


@pytest.fixture
def client(tmp_path):
    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "t",
            "DB_PATH": str(tmp_path / "t.db"),
            "PHOTO_DIR": str(tmp_path / "photos"),
            "RETRAIN_DIR": str(tmp_path / "retrain"),
            "VERIFIER": StubVerifier(),
            "PREDICT_THRESHOLD_MARGIN": 0.0,
        }
    )
    client = app.test_client()
    register(client, "alice", RED)
    register(client, "bobby", GREEN)
    login(client, "alice", RED)
    return client


def predict(client, color):
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buffer, "PNG")
    buffer.seek(0)
    return client.post("/api/predict", data={"photo": (buffer, "p.png")})


def test_predict_requires_login(client):
    client.post("/logout")
    assert predict(client, RED).status_code == 302


def test_predict_names_the_registered_person_in_the_photo(client):
    data = predict(client, GREEN).get_json()
    assert data["prediction"] == "bobby" and data["match"] is True
    assert data["candidates"][0]["username"] == "bobby"
    assert data["candidates"][0]["score"] > data["candidates"][1]["score"]


def test_predict_says_unknown_for_a_face_nobody_registered(client):
    data = predict(client, BLUE).get_json()
    assert data["prediction"] is None and data["match"] is False
    assert len(data["candidates"]) == 2  # still shows the closest registered people


def test_predict_without_a_face_gets_a_helpful_422(client):
    response = predict(client, BLACK)
    assert response.status_code == 422 and "one face" in response.get_json()["message"]


def test_predict_rejects_a_non_image(client):
    response = client.post("/api/predict", data={"photo": (io.BytesIO(b"nope"), "x.png")})
    assert response.status_code == 400


def test_predict_does_not_count_as_a_login_attempt(client):
    for _ in range(6):
        predict(client, BLUE)
    client.post("/logout")
    assert login(client, "alice", RED).status_code == 200  # no lockout from predictions


def make_client_with_margin(tmp_path, margin):
    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "t",
            "DB_PATH": str(tmp_path / "m.db"),
            "PHOTO_DIR": str(tmp_path / "photos"),
            "RETRAIN_DIR": str(tmp_path / "retrain"),
            "VERIFIER": StubVerifier(),  # login threshold 0.99
            "PREDICT_THRESHOLD_MARGIN": margin,
        }
    )
    client = app.test_client()
    register(client, "alice", RED)
    register(client, "bobby", GREEN)  # two registered people
    login(client, "alice", RED)
    return client


def test_predict_uses_a_stricter_bar_than_login(tmp_path):
    client = make_client_with_margin(tmp_path, 0.02)
    exact = predict(client, RED).get_json()
    assert exact["prediction"] == "alice"
    assert exact["threshold"] == pytest.approx(round(predict_bar(0.99, 2, 0.02), 3))
    assert exact["threshold"] > exact["login_threshold"] == pytest.approx(0.99)


def test_the_predict_bar_rises_with_more_registered_people(tmp_path):
    client = make_client_with_margin(tmp_path, 0.05)
    two = predict(client, RED).get_json()["threshold"]
    register(client, "carol", BLUE)
    register(client, "dave_", (255, 255, 0))
    register(client, "erin_", (0, 255, 255))
    assert predict(client, RED).get_json()["threshold"] > two


def test_a_face_between_the_two_bars_is_a_possible_match_not_a_name(tmp_path):
    client = make_client_with_margin(tmp_path, 0.032)  # bar for 2 people is about 0.998
    close = predict(client, (255, 20, 0)).get_json()  # similar to red: about 0.997
    assert close["prediction"] is None and close["match"] is False
    assert close["possible"] == "alice"
    assert close["candidates"][0]["score"] >= 0.99


def test_a_clearly_different_face_has_no_possible_match(tmp_path):
    client = make_client_with_margin(tmp_path, 0.02)
    far = predict(client, (0, 0, 255)).get_json()
    assert far["prediction"] is None and far["possible"] is None
