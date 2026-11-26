import json
import shutil
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, session

from app.auth import login_required
from app.db import get_db
from app.reenroll import reenroll_users
from src.retrain import IMAGE_EXTENSIONS, preprocess_report, validate_upload

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
    with closing(sqlite3.connect(app.config["DB_PATH"], timeout=10)) as conn:

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
            conn.execute("UPDATE uploaded_datasets SET status = 'used' WHERE job_id = ?", (job_id,))
        except Exception as error:  # report any failure on the dashboard
            conn.execute(
                "UPDATE retrain_jobs SET status = 'failed', message = ?, finished_at = ? WHERE id = ?",
                (str(error) or type(error).__name__, time.time(), job_id),
            )
            # a failed retrain must not consume the data: put it back in the queue
            conn.execute(
                "UPDATE uploaded_datasets SET status = 'saved', job_id = NULL WHERE job_id = ?", (job_id,)
            )
        conn.commit()


def _dataset_dict(row) -> dict:
    dataset = {k: row[k] for k in row.keys() if k != "folder"}
    dataset["report"] = json.loads(row["report"])
    return dataset


def _save_dataset(app, db, upload) -> tuple[dict | None, str | None]:
    """Validate an uploaded zip, keep its photos, and record it (with its preprocessing report)."""
    cursor = db.execute(
        "INSERT INTO uploaded_datasets (filename, status, people, photos, folder, report, created_at,"
        " uploaded_by) VALUES (?, 'saved', 0, 0, '', '{}', ?, ?)",
        (upload.filename or "upload.zip", time.time(), session["user"]),
    )
    db.commit()
    dataset_id = cursor.lastrowid
    folder = Path(app.config["DATASETS_DIR"]) / f"ds_{dataset_id}"
    folder.mkdir(parents=True, exist_ok=True)
    try:
        upload.save(folder / "upload.zip")
        data = validate_upload(str(folder / "upload.zip"), str(folder / "data"))
        report = preprocess_report(data)
    except ValueError as error:
        db.execute("DELETE FROM uploaded_datasets WHERE id = ?", (dataset_id,))
        db.commit()
        shutil.rmtree(folder, ignore_errors=True)
        return None, str(error)
    db.execute(
        "UPDATE uploaded_datasets SET people = ?, photos = ?, folder = ?, report = ? WHERE id = ?",
        (report["people"], report["photos"], str(folder), json.dumps(report), dataset_id),
    )
    db.commit()
    return _dataset_dict(db.execute("SELECT * FROM uploaded_datasets WHERE id = ?", (dataset_id,)).fetchone()), None


def _pending(db) -> list:
    return db.execute("SELECT * FROM uploaded_datasets WHERE status = 'saved' ORDER BY id").fetchall()


def _index_of(row) -> dict[str, list[str]]:
    """Rebuild person -> photos for a saved dataset (names prefixed so datasets never clash)."""
    index = {}
    for person in sorted((Path(row["folder"]) / "data").iterdir()):
        photos = sorted(str(p) for p in person.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if photos:
            index[f"ds{row['id']}_{person.name}"] = photos
    return index


def _start_job(app, db, mode: str, datasets: list):
    """Start a retrain job on the given saved datasets. Returns (job_id, error, status)."""
    with app.config["RETRAIN_LOCK"]:
        if db.execute("SELECT 1 FROM retrain_jobs WHERE status = 'running'").fetchone():
            return None, "A retrain job is already running. Your data stays saved for the next one.", 409
        cursor = db.execute(
            "INSERT INTO retrain_jobs (mode, status, progress, created_at, started_by)"
            " VALUES (?, 'running', 0, ?, ?)",
            (mode, time.time(), session.get("user", "auto-trigger")),
        )
        job_id = cursor.lastrowid
        db.executemany(
            "UPDATE uploaded_datasets SET status = 'training', job_id = ? WHERE id = ?",
            [(job_id, d["id"]) for d in datasets],
        )
        db.commit()
        job_dir = Path(app.config["RETRAIN_DIR"]) / f"job_{job_id}"
        job_dir.mkdir(parents=True, exist_ok=True)
        data = {}
        for dataset in datasets:
            data.update(_index_of(dataset))
        base_model = _active_model_path(db, app.config["MODEL_PATH"])
    threading.Thread(
        target=_work, args=(app, job_id, mode, data, job_dir, base_model), daemon=True
    ).start()
    return job_id, None, 202


@bp.post("/api/datasets")
@login_required
def upload_dataset():
    """Save a zip of faces (one folder per person) for later retraining."""
    app = current_app._get_current_object()
    db = get_db()
    upload = request.files.get("dataset")
    if upload is None:
        return _message(False, "Choose a zip file of faces (one folder per person).", 400)
    dataset, error = _save_dataset(app, db, upload)
    if error:
        return _message(False, error, 400)

    pending = _pending(db)
    auto_at = app.config["AUTO_RETRAIN_PEOPLE"]
    started = False
    if auto_at and sum(d["people"] for d in pending) >= auto_at:
        started = _start_job(app, db, "light", pending)[2] == 202
    return jsonify(ok=True, dataset=dataset, auto_retrain_started=started), 201


@bp.get("/api/datasets")
@login_required
def list_datasets():
    rows = get_db().execute("SELECT * FROM uploaded_datasets ORDER BY id DESC").fetchall()
    return jsonify(datasets=[_dataset_dict(r) for r in rows])


@bp.post("/api/retrain")
@login_required
def start():
    """Retrain on every saved-but-unused dataset (an attached zip is saved first)."""
    mode = request.form.get("mode", "")
    if mode not in ("light", "deep"):
        return _message(False, "Choose light or deep retraining.", 400)
    app = current_app._get_current_object()
    db = get_db()

    upload = request.files.get("dataset")
    if upload is not None:
        _, error = _save_dataset(app, db, upload)
        if error:
            return _message(False, error, 400)
    pending = _pending(db)
    if not pending:
        return _message(False, "Upload a dataset first (a zip with one folder per person).", 400)

    job_id, error, status = _start_job(app, db, mode, pending)
    if error:
        return _message(False, error, status)
    return _message(True, "Retraining started.", 202, job_id=job_id)


@bp.get("/api/retrain/status")
@login_required
def status():
    db = get_db()
    pending = _pending(db)
    people = sum(d["people"] for d in pending)
    return jsonify(
        job=_job_dict(_latest_job(db)),
        active_version=active_version(db),
        pending={"datasets": len(pending), "people": people, "photos": sum(d["photos"] for d in pending)},
        recommend_retrain=people >= current_app.config["RETRAIN_RECOMMEND_PEOPLE"],
    )


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
        threshold = result.get("login_threshold", {}).get("threshold", result["new"]["threshold"])
        new_verifier = app.config["VERIFIER_FACTORY"](job["model_path"], threshold)

        try:
            updated = reenroll_users(
                db.execute("SELECT id, username FROM users").fetchall(),
                app.config["PHOTO_DIR"],
                new_verifier,
            )
        except ValueError as error:
            return _message(False, str(error), 422)

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
