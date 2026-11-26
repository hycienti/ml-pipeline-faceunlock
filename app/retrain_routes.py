import json
import shutil
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path

import numpy as np
from flask import Blueprint, current_app, jsonify, request
from PIL import Image

from app.auth import login_required
from app.db import get_db
from src.retrain import validate_upload

bp = Blueprint("retrain", __name__)


def _message(ok: bool, text: str, status: int, **extra):
    return jsonify(ok=ok, message=text, **extra), status


def active_version(db) -> int:
    row = db.execute("SELECT MAX(version) FROM model_versions").fetchone()[0]
    return row or current_app.config["MODEL_VERSION"]


def _active_model_path(db, default_path: str) -> str:
    row = db.execute("SELECT path FROM model_versions ORDER BY version DESC LIMIT 1").fetchone()
    return row[0] if row else default_path


def _job_dict(row) -> dict | None:
    if row is None:
        return None
    job = dict(row)
    job["result"] = json.loads(job["result"]) if job["result"] else None
    job["promoted"] = bool(job["promoted"])
    job.pop("model_path", None)
    return job


def _latest_job(db):
    return db.execute("SELECT * FROM retrain_jobs ORDER BY id DESC LIMIT 1").fetchone()


def _work(app, job_id: int, mode: str, data: dict, job_dir: Path, base_model: str) -> None:
    with closing(sqlite3.connect(app.config["DB_PATH"])) as conn:

        def progress(fraction: float) -> None:
            conn.execute("UPDATE retrain_jobs SET progress = ? WHERE id = ?", (fraction, job_id))
            conn.commit()

        try:
            out_path = str(job_dir / "candidate.keras")
            result = app.config["RETRAIN_FN"](mode, data, base_model, out_path, progress)
            conn.execute(
                "UPDATE retrain_jobs SET status = 'done', progress = 1, result = ?, model_path = ?,"
                " finished_at = ? WHERE id = ?",
                (json.dumps(result), out_path, time.time(), job_id),
            )
        except Exception as error:  # report any failure on the dashboard
            conn.execute(
                "UPDATE retrain_jobs SET status = 'failed', message = ?, finished_at = ? WHERE id = ?",
                (str(error) or type(error).__name__, time.time(), job_id),
            )
        conn.commit()


@bp.post("/api/retrain")
@login_required
def start():
    from flask import session

    mode = request.form.get("mode", "")
    upload = request.files.get("dataset")
    if mode not in ("light", "deep"):
        return _message(False, "Choose light or deep retraining.", 400)
    if upload is None:
        return _message(False, "Choose a zip file of faces (one folder per person).", 400)

    app = current_app._get_current_object()
    db = get_db()
    with app.config["RETRAIN_LOCK"]:
        if db.execute("SELECT 1 FROM retrain_jobs WHERE status = 'running'").fetchone():
            return _message(False, "A retrain job is already running.", 409)
        cursor = db.execute(
            "INSERT INTO retrain_jobs (mode, status, progress, message, created_at, started_by)"
            " VALUES (?, 'running', 0, 'Validating upload', ?, ?)",
            (mode, time.time(), session["user"]),
        )
        db.commit()
        job_id = cursor.lastrowid
        job_dir = Path(app.config["RETRAIN_DIR"]) / f"job_{job_id}"
        job_dir.mkdir(parents=True, exist_ok=True)
        try:
            zip_path = job_dir / "upload.zip"
            upload.save(zip_path)
            data = validate_upload(str(zip_path), str(job_dir / "data"))
        except ValueError as error:
            db.execute("DELETE FROM retrain_jobs WHERE id = ?", (job_id,))
            db.commit()
            shutil.rmtree(job_dir, ignore_errors=True)
            return _message(False, str(error), 400)
        db.execute("UPDATE retrain_jobs SET message = NULL WHERE id = ?", (job_id,))
        db.commit()
        base_model = _active_model_path(db, app.config["MODEL_PATH"])

    threading.Thread(
        target=_work, args=(app, job_id, mode, data, job_dir, base_model), daemon=True
    ).start()
    return _message(True, "Retraining started.", 202, job_id=job_id)


@bp.get("/api/retrain/status")
@login_required
def status():
    db = get_db()
    return jsonify(job=_job_dict(_latest_job(db)), active_version=active_version(db))


@bp.post("/api/retrain/promote")
@login_required
def promote():
    app = current_app._get_current_object()
    db = get_db()
    with app.config["PROMOTE_LOCK"]:
        job = _latest_job(db)
        if job is None or job["status"] != "done" or job["promoted"]:
            return _message(False, "There is no finished, unpromoted retrain to promote.", 409)

        result = json.loads(job["result"])
        threshold = result["new"]["threshold"]
        new_verifier = app.config["VERIFIER_FACTORY"](job["model_path"], threshold)

        updated = []
        for user in db.execute("SELECT id, username FROM users").fetchall():
            folder = Path(app.config["PHOTO_DIR"]) / user["username"].lower()
            photos = sorted(folder.glob("*.jpg"))
            try:
                if not photos:
                    raise ValueError("no enrollment photos found")
                images = [np.asarray(Image.open(p).convert("RGB")) for p in photos]
                embedding = new_verifier.enroll(images)
            except ValueError as error:
                return _message(
                    False,
                    f"Could not re-enroll {user['username']} with the new model: {error}",
                    422,
                )
            updated.append((embedding.astype(np.float32).tobytes(), user["id"]))

        version = active_version(db) + 1
        target = Path(app.config["ACTIVE_MODELS_DIR"]) / f"model_v{version}.keras"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(job["model_path"], target)
        try:
            db.execute("BEGIN")
            db.executemany(
                f"UPDATE users SET embedding = ?, model_version = {version} WHERE id = ?", updated
            )
            db.execute(
                "INSERT INTO model_versions (version, path, threshold, created_at, job_id)"
                " VALUES (?, ?, ?, ?, ?)",
                (version, str(target), threshold, time.time(), job["id"]),
            )
            db.execute("UPDATE retrain_jobs SET promoted = 1 WHERE id = ?", (job["id"],))
            db.commit()
        except Exception:
            db.rollback()
            raise
        app.config["VERIFIER"] = new_verifier
        app.config["MODEL_VERSION"] = version
    return _message(True, f"Model v{version} is now live.", 200, version=version)
