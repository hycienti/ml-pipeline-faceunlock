import io

import numpy as np
import pytest
from PIL import Image

from app import create_app
from src.prediction import FaceVerifier

RED, GREEN, BLACK = (255, 0, 0), (0, 255, 0), (0, 0, 0)


class StubVerifier(FaceVerifier):
    """Treats an image's top-left pixel colour as its 'face'. Black means no face."""

    def __init__(self):
        super().__init__(model=None, threshold=0.99)

    def embed(self, image):
        color = image[0, 0].astype(float)
        if color.sum() == 0:
            return None
        return color / np.linalg.norm(color)

    def embed_many(self, images):
        return [self.embed(image) for image in images]


def png(color):
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


@pytest.fixture
def client(tmp_path):
    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test",
            "DB_PATH": str(tmp_path / "test.db"),
            "PHOTO_DIR": str(tmp_path / "photos"),
            "VERIFIER": StubVerifier(),
        }
    )
    return app.test_client()


def register(client, username, color=RED, n=3):
    photos = [(png(color), f"{i}.png") for i in range(n)]
    return client.post("/api/register", data={"username": username, "photos": photos})


def login(client, username, color=RED):
    return client.post(
        "/api/login", data={"username": username, "photo": (png(color), "p.png")}
    )


def test_register_creates_user_and_saves_photos(client, tmp_path):
    response = register(client, "alice")
    assert response.status_code == 201
    assert len(list((tmp_path / "photos" / "alice").glob("*.jpg"))) == 3


@pytest.mark.parametrize("username", ["", "  ", "ab", "has space", "x" * 31, "bad/name"])
def test_register_rejects_invalid_usernames(client, username):
    assert register(client, username).status_code == 400


def test_register_rejects_duplicate_and_case_variant(client):
    assert register(client, "alice").status_code == 201
    assert register(client, "alice").status_code == 409
    assert register(client, "ALICE").status_code == 409
    assert register(client, " alice ").status_code == 409


def test_register_requires_exactly_three_photos(client):
    assert register(client, "alice", n=2).status_code == 400


def test_register_rejects_photo_without_a_face_and_creates_no_user(client):
    assert register(client, "alice", color=BLACK).status_code == 400
    assert register(client, "alice").status_code == 201


def test_register_rejects_non_image_upload(client):
    data = {"username": "alice", "photos": [(io.BytesIO(b"not an image"), "x.png")] * 3}
    assert client.post("/api/register", data=data).status_code == 400


def test_login_with_own_face_opens_dashboard(client):
    register(client, "alice")
    assert login(client, "alice").status_code == 200
    page = client.get("/dashboard")
    assert page.status_code == 200 and b"alice" in page.data


def test_login_is_case_insensitive_on_username(client):
    register(client, "alice")
    assert login(client, "ALICE").status_code == 200


def test_wrong_face_and_unknown_user_fail_identically(client):
    register(client, "alice")
    wrong_face = login(client, "alice", color=GREEN)
    unknown = login(client, "nobody")
    assert wrong_face.status_code == unknown.status_code == 401
    assert wrong_face.get_json() == unknown.get_json()


def test_dashboard_requires_login(client):
    response = client.get("/dashboard")
    assert response.status_code == 302 and "/login" in response.headers["Location"]


def test_logout_ends_the_session(client):
    register(client, "alice")
    login(client, "alice")
    client.post("/logout")
    assert client.get("/dashboard").status_code == 302


def test_lockout_after_five_failures_even_with_correct_face(client):
    register(client, "alice")
    for _ in range(5):
        assert login(client, "alice", color=GREEN).status_code == 401
    assert login(client, "alice", color=RED).status_code == 429


def test_lockout_applies_to_unknown_usernames_too(client):
    for _ in range(5):
        assert login(client, "ghost").status_code == 401
    assert login(client, "ghost").status_code == 429


def test_photo_without_a_face_gets_a_helpful_message_and_does_not_count_toward_lockout(client):
    register(client, "alice")
    for _ in range(7):
        response = login(client, "alice", color=BLACK)  # the stub finds no face in black
        assert response.status_code == 422
        assert "one face" in response.get_json()["message"]
    assert login(client, "alice", color=RED).status_code == 200  # not locked out


def test_no_face_message_is_the_same_for_unknown_users(client):
    register(client, "alice")
    known = login(client, "alice", color=BLACK)
    unknown = login(client, "nobody", color=BLACK)
    assert known.status_code == unknown.status_code == 422
    assert known.get_json() == unknown.get_json()


def test_locked_users_stay_locked_even_with_a_no_face_photo(client):
    register(client, "alice")
    for _ in range(5):
        login(client, "alice", color=GREEN)
    assert login(client, "alice", color=BLACK).status_code == 429
