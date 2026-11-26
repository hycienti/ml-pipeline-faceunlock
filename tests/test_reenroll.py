import numpy as np
import pytest

from app import create_app
from app.db import get_db
from app.reenroll import reenroll_users
from tests.test_app_auth import RED, StubVerifier, register


class RotatedStub(StubVerifier):
    """A 'new model': same faces, different embedding space."""

    def embed(self, image):
        embedding = super().embed(image)
        return None if embedding is None else np.roll(embedding, 1)


@pytest.fixture
def app(tmp_path):
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
    return app


def test_every_user_gets_a_new_embedding_from_their_stored_photos(app):
    with app.app_context():
        users = get_db().execute("SELECT id, username, embedding FROM users").fetchall()
        updated = reenroll_users(users, app.config["PHOTO_DIR"], RotatedStub())
    assert len(updated) == 1 and updated[0][1] == users[0]["id"]
    old = np.frombuffer(users[0]["embedding"], dtype=np.float32)
    new = np.frombuffer(updated[0][0], dtype=np.float32)
    assert np.allclose(new, np.roll(old, 1), atol=1e-2)  # JPEG round trip tolerance
    assert not np.allclose(new, old, atol=1e-2)


def test_a_user_without_photos_stops_everything_and_is_named(app, tmp_path):
    for photo in (tmp_path / "photos" / "alice").glob("*.jpg"):
        photo.unlink()
    with app.app_context():
        users = get_db().execute("SELECT id, username FROM users").fetchall()
        with pytest.raises(ValueError, match="alice"):
            reenroll_users(users, app.config["PHOTO_DIR"], RotatedStub())
