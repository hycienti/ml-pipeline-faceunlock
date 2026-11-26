import io

import pytest
from PIL import Image

from app import create_app
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
