"""Deploy FaceUnlock on Modal.

    pip install modal
    python3 -m modal setup          # one-time browser login
    modal deploy modal_app.py       # prints the public https URL

How it is run (and why):
* ONE container (max_containers=1). The database is SQLite, which wants a single writer.
* It scales to zero when idle and wakes on the next request. The dashboard counts that idle
  time as "up" because the app records a clean stop when it sleeps (see app/heartbeat.py).
* Users, photos, uploaded datasets, promoted models and the session key live on a Modal Volume
  (mounted at /data). The volume is committed every 30 s and at shutdown, after folding the
  database's write-ahead log into the main file so the snapshot is consistent.
"""
import sqlite3
import threading
from pathlib import Path

import modal

DATA = "/data"
INSTANCE = f"{DATA}/instance"
volume = modal.Volume.from_name("faceunlock-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libglib2.0-0", "libgl1")
    .pip_install_from_requirements("requirements-app.txt")
    .env({"FACEUNLOCK_INSTANCE": INSTANCE, "FACEUNLOCK_PLATFORM": "Modal", "TF_CPP_MIN_LOG_LEVEL": "2", "PYTHONUNBUFFERED": "1"})
    .add_local_dir("app", "/root/app", ignore=["**/__pycache__"])
    .add_local_dir("src", "/root/src", ignore=["**/__pycache__"])
    .add_local_dir("models", "/root/models", ignore=["*.h5", "*.ipynb"])
)

app = modal.App("faceunlock")


def commit_volume() -> None:
    """Fold the SQLite write-ahead log into the database file, then persist the volume."""
    db = Path(INSTANCE) / "faceunlock.db"
    if db.exists():
        connection = sqlite3.connect(db, timeout=10)
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.close()
    volume.commit()


@app.cls(
    image=image,
    volumes={DATA: volume},
    cpu=2.0,
    memory=4096,
    max_containers=1,
    scaledown_window=300,  # seconds idle before sleeping
    timeout=3600,
)
@modal.concurrent(max_inputs=8)
class FaceUnlock:
    @modal.enter()
    def start(self):
        from app import create_app

        Path(INSTANCE).mkdir(parents=True, exist_ok=True)
        self.flask = create_app()  # starts the heartbeat and the dashboard warm-up thread
        self.stopping = threading.Event()
        threading.Thread(target=self.commit_forever, daemon=True).start()

    def commit_forever(self):
        while not self.stopping.wait(30):
            commit_volume()

    @modal.wsgi_app()
    def web(self):
        return self.flask

    @modal.exit()
    def stop(self):
        from app import heartbeat

        self.stopping.set()
        heartbeat.record_stop(self.flask)  # a clean stop: the sleep that follows counts as up
        commit_volume()
