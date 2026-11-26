import json
import sqlite3

import pytest

from app import create_app
from app.dashboard import uptime_summary
from app.db import SCHEMA
from tests.test_app_auth import RED, StubVerifier, login, register


@pytest.fixture
def make_client(tmp_path):
    def build(**config):
        app = create_app(
            {
                "TESTING": True,
                "SECRET_KEY": "test",
                "DB_PATH": str(tmp_path / "test.db"),
                "PHOTO_DIR": str(tmp_path / "photos"),
                "VERIFIER": StubVerifier(),
                "METRICS_PATH": str(tmp_path / "metrics.json"),
                "DATASET_STATS_PATH": str(tmp_path / "dataset_stats.json"),
                **config,
            }
        )
        client = app.test_client()
        register(client, "alice")
        login(client, "alice", RED)
        return client

    return build


def memory_db(heartbeats, starts=()):
    db = sqlite3.connect(":memory:")
    db.executescript(SCHEMA)
    db.executemany("INSERT INTO heartbeats (ts) VALUES (?)", [(t,) for t in heartbeats])
    db.executemany("INSERT INTO app_starts (ts) VALUES (?)", [(t,) for t in starts])
    return db


def test_uptime_summary_buckets_and_running_since():
    now = 1_000_000.0
    db = memory_db([now - 4000, now - 60], starts=[now - 4000, now - 500])
    summary = uptime_summary(db, now=now)
    assert len(summary["buckets"]) == 288
    assert summary["buckets"][0]["up"] is None  # before the first heartbeat
    assert summary["buckets"][-1]["up"] == 1
    assert summary["running_since"] == now - 500
    assert summary["uptime_seconds"] == 500
    assert 0 < summary["uptime_pct"] <= 100


def test_uptime_summary_without_data():
    summary = uptime_summary(memory_db([]), now=1_000_000.0)
    assert summary["uptime_pct"] is None and summary["running_since"] is None


def test_api_requires_login(make_client):
    client = make_client()
    client.post("/logout")
    for path in ["/api/uptime", "/api/metrics", "/api/dataset-stats"]:
        assert client.get(path).status_code == 302


def test_uptime_endpoint_reports_deployment_info(make_client):
    data = make_client().get("/api/uptime").get_json()
    assert data["running_since"] is not None
    assert data["deployment"]["model_version"] == 1
    assert "platform" in data["deployment"]


def test_metrics_endpoint_unavailable_then_available(make_client, tmp_path):
    assert make_client().get("/api/metrics").get_json() == {"available": False}
    report = {"run": "4 + fine-tuning", "threshold": 0.5, "metrics": {"test": {"auc": 0.9}}}
    (tmp_path / "metrics.json").write_text(json.dumps(report))
    data = make_client().get("/api/metrics").get_json()
    assert data["available"] is True and data["run"] == "4 + fine-tuning"


def test_dataset_stats_endpoint(make_client, tmp_path):
    assert make_client().get("/api/dataset-stats").get_json() == {"available": False}
    (tmp_path / "dataset_stats.json").write_text(json.dumps({"splits": {"train": 1}}))
    data = make_client().get("/api/dataset-stats").get_json()
    assert data["available"] is True and data["splits"] == {"train": 1}
