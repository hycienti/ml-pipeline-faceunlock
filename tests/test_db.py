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
