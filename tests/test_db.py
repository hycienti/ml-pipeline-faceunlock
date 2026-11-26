import sqlite3
import threading

from app import create_app
from app.db import get_db
from tests.test_app_auth import StubVerifier


def make_app(tmp_path):
    return create_app(
        {
            "TESTING": True,
            "DB_PATH": str(tmp_path / "t.db"),
            "PHOTO_DIR": str(tmp_path / "photos"),
            "RETRAIN_DIR": str(tmp_path / "retrain"),
            "VERIFIER": StubVerifier(),
        }
    )


def test_database_uses_wal_mode_and_waits_for_locks(tmp_path):
    app = make_app(tmp_path)
    with app.app_context():
        db = get_db()
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert db.execute("PRAGMA busy_timeout").fetchone()[0] >= 5000


def test_several_writers_do_not_lock_each_other_out(tmp_path):
    app = make_app(tmp_path)
    errors = []

    def write(n):
        try:
            conn = sqlite3.connect(app.config["DB_PATH"], timeout=10)
            for i in range(30):
                conn.execute("INSERT INTO heartbeats (ts) VALUES (?)", (n * 1000 + i,))
                conn.commit()
            conn.close()
        except sqlite3.OperationalError as error:  # "database is locked"
            errors.append(error)

    threads = [threading.Thread(target=write, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    with app.app_context():
        assert get_db().execute("SELECT COUNT(*) FROM heartbeats").fetchone()[0] == 180


def test_instance_folder_can_be_moved_to_persistent_storage_with_an_environment_variable(tmp_path, monkeypatch):
    monkeypatch.setenv("FACEUNLOCK_INSTANCE", str(tmp_path / "persistent"))
    app = create_app({"TESTING": True, "VERIFIER": StubVerifier()})
    assert app.instance_path == str(tmp_path / "persistent")
    for key in ("DB_PATH", "PHOTO_DIR", "RETRAIN_DIR", "DATASETS_DIR", "REPLAY_DIR", "INSIGHT_DIR", "ACTIVE_MODELS_DIR"):
        assert app.config[key].startswith(str(tmp_path / "persistent")), key
    assert (tmp_path / "persistent" / "faceunlock.db").exists()


def fresh_app(tmp_path, **extra):
    return create_app(
        {
            "TESTING": True,
            "DB_PATH": str(tmp_path / "t.db"),
            "PHOTO_DIR": str(tmp_path / "photos"),
            "RETRAIN_DIR": str(tmp_path / "retrain"),
            "VERIFIER": StubVerifier(),
            **extra,
        }
    )


def test_session_key_is_random_persisted_and_shared_between_restarts(tmp_path, monkeypatch):
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.setenv("FACEUNLOCK_INSTANCE", str(tmp_path / "inst"))
    first = fresh_app(tmp_path).config["SECRET_KEY"]
    key_file = tmp_path / "inst" / "secret_key"
    assert key_file.read_text().strip() == first and len(first) == 64
    assert key_file.stat().st_mode & 0o777 == 0o600
    assert first != "dev-only-change-me"
    assert fresh_app(tmp_path).config["SECRET_KEY"] == first  # a restart keeps sessions valid


def test_secret_key_environment_variable_wins_and_writes_no_file(tmp_path, monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "from-the-environment")
    monkeypatch.setenv("FACEUNLOCK_INSTANCE", str(tmp_path / "inst"))
    assert fresh_app(tmp_path).config["SECRET_KEY"] == "from-the-environment"
    assert not (tmp_path / "inst" / "secret_key").exists()


def test_retrain_trigger_settings_can_come_from_the_environment(tmp_path, monkeypatch):
    assert fresh_app(tmp_path).config["AUTO_RETRAIN_PEOPLE"] == 0  # off unless asked for
    monkeypatch.setenv("AUTO_RETRAIN_PEOPLE", "15")
    monkeypatch.setenv("RETRAIN_RECOMMEND_PEOPLE", "5")
    app = fresh_app(tmp_path)
    assert app.config["AUTO_RETRAIN_PEOPLE"] == 15 and app.config["RETRAIN_RECOMMEND_PEOPLE"] == 5
