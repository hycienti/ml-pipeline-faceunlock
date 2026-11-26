import io
import threading
import time

import pytest

from app import create_app
from tests.test_app_auth import RED, StubVerifier, login, register
from tests.test_retrain import make_zip, ten_people
from tests.test_retrain_api import FAKE_RESULT, instant_retrain, wait_for


@pytest.fixture
def make_client(tmp_path):
    def build(**config):
        app = create_app(
            {
                "TESTING": True,
                "SECRET_KEY": "t",
                "DB_PATH": str(tmp_path / "t.db"),
                "PHOTO_DIR": str(tmp_path / "photos"),
                "RETRAIN_DIR": str(tmp_path / "retrain"),
                "DATASETS_DIR": str(tmp_path / "datasets"),
                "ACTIVE_MODELS_DIR": str(tmp_path / "models"),
                "VERIFIER": StubVerifier(),
                "VERIFIER_FACTORY": lambda path, threshold: StubVerifier(),
                "RETRAIN_FN": instant_retrain,
                **config,
            }
        )
        client = app.test_client()
        register(client, "alice", RED)
        login(client, "alice", RED)
        return client

    return build


def post_zip(client, tmp_path, url="/api/datasets", name="up.zip", layout=None, **form):
    zip_path = make_zip(tmp_path / name, layout or ten_people())
    return client.post(url, data={**form, "dataset": (io.BytesIO(zip_path.read_bytes()), name)})


def test_uploading_a_dataset_saves_it_with_a_preprocessing_report(make_client, tmp_path):
    client = make_client()
    response = post_zip(client, tmp_path)
    assert response.status_code == 201
    dataset = response.get_json()["dataset"]
    assert dataset["people"] == 10 and dataset["photos"] == 20 and dataset["status"] == "saved"
    assert {"face_found_pct", "center_crop_fallback", "median_size"} <= set(dataset["report"])
    listed = client.get("/api/datasets").get_json()["datasets"]
    assert [d["id"] for d in listed] == [dataset["id"]] and listed[0]["filename"] == "up.zip"


def test_a_bad_upload_saves_nothing(make_client):
    client = make_client()
    response = client.post("/api/datasets", data={"dataset": (io.BytesIO(b"nope"), "x.zip")})
    assert response.status_code == 400 and "valid zip" in response.get_json()["message"]
    assert client.get("/api/datasets").get_json()["datasets"] == []


def test_retrain_with_nothing_uploaded_asks_for_data(make_client):
    response = make_client().post("/api/retrain", data={"mode": "light"})
    assert response.status_code == 400 and "upload" in response.get_json()["message"].lower()


def test_retrain_uses_the_saved_datasets_and_marks_them_used(make_client, tmp_path):
    client = make_client()
    dataset_id = post_zip(client, tmp_path).get_json()["dataset"]["id"]
    assert client.post("/api/retrain", data={"mode": "light"}).status_code == 202
    job = wait_for(client, "done")
    state = client.get("/api/retrain/status").get_json()
    used = next(d for d in client.get("/api/datasets").get_json()["datasets"] if d["id"] == dataset_id)
    assert used["status"] == "used" and used["job_id"] == job["id"]
    assert state["pending"] == {"datasets": 0, "people": 0, "photos": 0}


def test_two_saved_datasets_are_trained_together_without_name_clashes(make_client, tmp_path):
    seen = {}

    def spy(mode, data, base, out, progress):
        seen["people"] = sorted(data)
        return instant_retrain(mode, data, base, out, progress)

    client = make_client(RETRAIN_FN=spy)
    post_zip(client, tmp_path, name="a.zip")
    post_zip(client, tmp_path, name="b.zip")  # same person names as the first
    client.post("/api/retrain", data={"mode": "light"})
    wait_for(client, "done")
    assert len(seen["people"]) == 20  # 10 + 10, none overwritten


def test_status_reports_pending_data_and_recommends_retraining(make_client, tmp_path):
    client = make_client()
    assert client.get("/api/retrain/status").get_json()["recommend_retrain"] is False
    post_zip(client, tmp_path)
    state = client.get("/api/retrain/status").get_json()
    assert state["pending"] == {"datasets": 1, "people": 10, "photos": 20}
    assert state["recommend_retrain"] is True


def test_one_step_upload_and_retrain_still_works(make_client, tmp_path):
    client = make_client()
    assert post_zip(client, tmp_path, url="/api/retrain", mode="deep").status_code == 202
    assert wait_for(client, "done")["mode"] == "deep"
    assert len(client.get("/api/datasets").get_json()["datasets"]) == 1


def test_automatic_trigger_starts_a_light_retrain_when_enough_data_arrives(make_client, tmp_path):
    client = make_client(AUTO_RETRAIN_PEOPLE=10)
    response = post_zip(client, tmp_path)
    assert response.get_json()["auto_retrain_started"] is True
    assert wait_for(client, "done")["mode"] == "light"


def test_automatic_trigger_is_off_by_default_and_waits_for_enough_people(make_client, tmp_path):
    client = make_client()
    assert post_zip(client, tmp_path).get_json()["auto_retrain_started"] is False
    assert client.get("/api/retrain/status").get_json()["job"] is None
    client = make_client(AUTO_RETRAIN_PEOPLE=50)
    assert post_zip(client, tmp_path, name="c.zip").get_json()["auto_retrain_started"] is False


def test_dataset_endpoints_require_login(make_client):
    client = make_client()
    client.post("/logout")
    assert client.get("/api/datasets").status_code == 302
    assert client.post("/api/datasets").status_code == 302
