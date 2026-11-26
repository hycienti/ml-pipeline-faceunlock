import io
import threading
import time

import pytest

from app import create_app
from app.db import get_db
from tests.test_app_auth import GREEN, RED, StubVerifier, login, register
from tests.test_retrain import make_zip, ten_people

FAKE_RESULT = {
    "mode": "light",
    "people_train": 7,
    "people_eval": 3,
    "eval_pairs": 400,
    "old": {"auc": 0.70, "threshold": 0.5},
    "new": {"auc": 0.80, "threshold": 0.61},
    "login_threshold": {"threshold": 0.47, "far_target": 0.02, "far": 0.02, "frr": 0.3},
}


def build_app(tmp_path, **config):
    return create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test",
            "DB_PATH": str(tmp_path / "test.db"),
            "PHOTO_DIR": str(tmp_path / "photos"),
            "RETRAIN_DIR": str(tmp_path / "retrain"),
            "ACTIVE_MODELS_DIR": str(tmp_path / "models"),
            "VERIFIER": StubVerifier(),
            "VERIFIER_FACTORY": lambda path, threshold: StubVerifier(),
            "RETRAIN_FN": instant_retrain,
            **config,
        }
    )


def instant_retrain(mode, data, base_model_path, out_path, progress):
    progress(0.5)
    open(out_path, "wb").write(b"fake model")
    return {**FAKE_RESULT, "mode": mode}


@pytest.fixture
def make_client(tmp_path):
    def build(**config):
        app = build_app(tmp_path, **config)
        client = app.test_client()
        register(client, "alice")
        login(client, "alice", RED)
        return client

    return build


def upload(client, tmp_path, mode="light", layout=None):
    zip_path = make_zip(tmp_path / "up.zip", layout or ten_people())
    with open(zip_path, "rb") as handle:
        return client.post(
            "/api/retrain", data={"mode": mode, "dataset": (io.BytesIO(handle.read()), "up.zip")}
        )


def wait_for(client, status, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get("/api/retrain/status").get_json()["job"]
        if job and job["status"] == status:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job never reached {status}: {job}")


def test_retrain_requires_login(make_client):
    client = make_client()
    client.post("/logout")
    assert client.post("/api/retrain").status_code == 302
    assert client.get("/api/retrain/status").status_code == 302


def test_bad_mode_and_missing_file_are_rejected(make_client, tmp_path):
    client = make_client()
    assert upload(client, tmp_path, mode="medium").status_code == 400
    assert client.post("/api/retrain", data={"mode": "light"}).status_code == 400


def test_non_zip_upload_is_rejected_and_creates_no_job(make_client):
    client = make_client()
    response = client.post(
        "/api/retrain", data={"mode": "light", "dataset": (io.BytesIO(b"nope"), "x.zip")}
    )
    assert response.status_code == 400 and "valid zip" in response.get_json()["message"]
    assert client.get("/api/retrain/status").get_json()["job"] is None


def test_successful_job_reports_result(make_client, tmp_path):
    client = make_client()
    assert upload(client, tmp_path, mode="deep").status_code == 202
    job = wait_for(client, "done")
    assert job["mode"] == "deep" and job["progress"] == 1
    assert job["result"]["new"]["auc"] == 0.80 and job["promoted"] is False


def test_second_job_is_refused_while_one_runs(make_client, tmp_path):
    release = threading.Event()

    def slow(mode, data, base, out, progress):
        release.wait(5)
        return instant_retrain(mode, data, base, out, progress)

    client = make_client(RETRAIN_FN=slow)
    assert upload(client, tmp_path).status_code == 202
    assert upload(client, tmp_path).status_code == 409
    release.set()
    wait_for(client, "done")
    assert upload(client, tmp_path).status_code == 202
    wait_for(client, "done")


def test_failed_job_reports_the_reason(make_client, tmp_path):
    def boom(*args, **kwargs):
        raise RuntimeError("out of memory")

    client = make_client(RETRAIN_FN=boom)
    upload(client, tmp_path)
    job = wait_for(client, "failed")
    assert "out of memory" in job["message"]


def test_job_left_running_by_a_restart_is_marked_failed(tmp_path):
    app = build_app(tmp_path)
    with app.app_context():
        db = get_db()
        db.execute(
            "INSERT INTO retrain_jobs (mode, status, progress, created_at) VALUES ('light', 'running', 0.3, 0)"
        )
        db.commit()
    app = build_app(tmp_path)
    client = app.test_client()
    register(client, "alice")
    login(client, "alice")
    job = client.get("/api/retrain/status").get_json()["job"]
    assert job["status"] == "failed" and "restart" in job["message"].lower()


def test_promote_needs_a_finished_unpromoted_job(make_client, tmp_path):
    client = make_client()
    assert client.post("/api/retrain/promote").status_code == 409
    upload(client, tmp_path)
    wait_for(client, "done")
    assert client.post("/api/retrain/promote").status_code == 200
    assert client.post("/api/retrain/promote").status_code == 409


def test_promote_re_embeds_users_and_bumps_the_version(make_client, tmp_path):
    client = make_client()
    upload(client, tmp_path)
    wait_for(client, "done")
    data = client.post("/api/retrain/promote").get_json()
    assert data["ok"] and data["version"] == 2
    assert client.get("/api/retrain/status").get_json()["active_version"] == 2
    assert (tmp_path / "models" / "model_v2.keras").exists()
    client.post("/logout")
    assert login(client, "alice", RED).status_code == 200
    assert login(client, "alice", GREEN).status_code == 401


def test_promote_is_all_or_nothing_when_a_user_cannot_be_re_embedded(make_client, tmp_path):
    client = make_client()
    upload(client, tmp_path)
    wait_for(client, "done")
    for photo in (tmp_path / "photos" / "alice").glob("*.jpg"):
        photo.unlink()
    response = client.post("/api/retrain/promote")
    assert response.status_code == 422 and "alice" in response.get_json()["message"]
    assert client.get("/api/retrain/status").get_json()["active_version"] == 1
    assert client.get("/api/retrain/status").get_json()["job"]["promoted"] is False


def test_default_retrain_uses_the_replay_set_when_it_ships_with_the_app(tmp_path):
    app = create_app(
        {
            "TESTING": True,
            "DB_PATH": str(tmp_path / "t.db"),
            "PHOTO_DIR": str(tmp_path / "photos"),
            "RETRAIN_DIR": str(tmp_path / "retrain"),
            "VERIFIER": StubVerifier(),
        }
    )
    assert app.config["RETRAIN_FN"].keywords["replay_zip"].endswith("replay_faces.zip")


def test_promote_uses_the_login_threshold_not_the_noisy_eer(make_client, tmp_path):
    client = make_client()
    upload(client, tmp_path)
    wait_for(client, "done")
    client.post("/api/retrain/promote")
    app = client.application
    with app.app_context():
        threshold = get_db().execute("SELECT threshold FROM model_versions").fetchone()[0]
    assert threshold == 0.47
