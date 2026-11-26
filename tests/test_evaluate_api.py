import pytest

from app import create_app
from tests.test_app_auth import GREEN, RED, StubVerifier, login, register
from tests.test_replay import make_replay_zip


@pytest.fixture
def make_client(tmp_path):
    def build(replay=True):
        app = create_app(
            {
                "TESTING": True,
                "SECRET_KEY": "t",
                "DB_PATH": str(tmp_path / "t.db"),
                "PHOTO_DIR": str(tmp_path / "photos"),
                "RETRAIN_DIR": str(tmp_path / "retrain"),
                "REPLAY_DIR": str(tmp_path / "replay"),
                "REPLAY_ZIP": str(make_replay_zip(tmp_path / "replay.zip", 6, 6))
                if replay
                else str(tmp_path / "none.zip"),
                "VERIFIER": StubVerifier(),
            }
        )
        client = app.test_client()
        register(client, "alice", RED)
        login(client, "alice", RED)
        return client

    return build


def test_evaluation_requires_login(make_client):
    client = make_client()
    client.post("/logout")
    assert client.get("/api/evaluate").status_code == 302
    assert client.post("/api/evaluate").status_code == 302


def test_before_any_run_there_is_no_evaluation_but_traffic_is_reported(make_client):
    data = make_client().get("/api/evaluate").get_json()
    assert data["latest"] is None and data["history"] == []
    assert data["traffic"]["attempts"] == 1 and data["traffic"]["passes"] == 1


def test_running_an_evaluation_stores_it_and_returns_the_metrics(make_client):
    client = make_client()
    run = client.post("/api/evaluate").get_json()
    assert run["ok"] and run["latest"]["model_version"] == 1
    assert {"accuracy", "f1", "auc"} <= set(run["latest"]["metrics"])
    again = client.get("/api/evaluate").get_json()
    assert again["latest"]["id"] == run["latest"]["id"] and len(again["history"]) == 1
    client.post("/api/evaluate")
    assert len(client.get("/api/evaluate").get_json()["history"]) == 2


def test_evaluation_without_the_replay_set_explains_what_is_missing(make_client):
    response = make_client(replay=False).post("/api/evaluate")
    assert response.status_code == 503 and "replay" in response.get_json()["message"].lower()


def test_traffic_counts_passes_and_failures(make_client):
    client = make_client()
    login(client, "alice", GREEN)  # one failure
    traffic = client.get("/api/evaluate").get_json()["traffic"]
    assert traffic["attempts"] == 2 and traffic["passes"] == 1 and traffic["pass_rate"] == 0.5
