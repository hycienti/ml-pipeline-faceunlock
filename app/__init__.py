import json
import os
import threading
import time
from pathlib import Path

from flask import Flask

from app import auth, dashboard, db, heartbeat, retrain_routes, views

ROOT = Path(__file__).resolve().parent.parent


def run_retrain_job(*args, **kwargs):
    from src.retrain import run_retrain

    return run_retrain(*args, **kwargs)


def verifier_from_files(model_path: str, threshold: float):
    from src.prediction import FaceVerifier

    return FaceVerifier.from_path(model_path, threshold)


def load_verifier(app: Flask):
    """Load the promoted model if there is one, otherwise the notebook's model."""
    with app.app_context():
        active = db.get_db().execute(
            "SELECT version, path, threshold FROM model_versions ORDER BY version DESC LIMIT 1"
        ).fetchone()
    if active:
        app.config["MODEL_VERSION"] = active["version"]
        return app.config["VERIFIER_FACTORY"](active["path"], active["threshold"])
    metrics = json.loads(Path(app.config["METRICS_PATH"]).read_text())
    return app.config["VERIFIER_FACTORY"](app.config["MODEL_PATH"], metrics["threshold"])


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__, instance_path=str(ROOT / "instance"))
    app.config.update(
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev-only-change-me"),
        DB_PATH=str(Path(app.instance_path) / "faceunlock.db"),
        PHOTO_DIR=str(Path(app.instance_path) / "photos"),
        MODEL_PATH=str(ROOT / "models" / "faceunlock_v1.keras"),
        METRICS_PATH=str(ROOT / "models" / "metrics_v1.json"),
        DATASET_STATS_PATH=str(ROOT / "models" / "dataset_stats.json"),
        SAMPLE_FACES_PATH=str(ROOT / "models" / "sample_faces.jpg"),
        RETRAIN_DIR=str(Path(app.instance_path) / "retrain"),
        ACTIVE_MODELS_DIR=str(Path(app.instance_path) / "models"),
        VERIFIER_FACTORY=verifier_from_files,
        HEARTBEAT_SECONDS=60,
        MODEL_VERSION=1,
        MAX_FAILED_LOGINS=5,
        LOCKOUT_SECONDS=300,
        MAX_CONTENT_LENGTH=512 * 1024 * 1024,  # retrain zips can be large
    )
    if test_config:
        app.config.update(test_config)

    Path(app.config["DB_PATH"]).parent.mkdir(parents=True, exist_ok=True)
    Path(app.config["PHOTO_DIR"]).mkdir(parents=True, exist_ok=True)
    app.config.setdefault("RETRAIN_FN", run_retrain_job)
    app.config["RETRAIN_LOCK"] = threading.Lock()
    app.config["PROMOTE_LOCK"] = threading.Lock()
    Path(app.config["RETRAIN_DIR"]).mkdir(parents=True, exist_ok=True)

    app.teardown_appcontext(db.close_db)
    with app.app_context():
        db.init_db()
        db.get_db().execute(
            "UPDATE retrain_jobs SET status = 'failed', message = 'Interrupted by a restart.',"
            " finished_at = ? WHERE status = 'running'",
            (time.time(),),
        )
        db.get_db().commit()
    if "VERIFIER" not in app.config:
        app.config["VERIFIER"] = load_verifier(app)
    with app.app_context():
        db.get_db().execute("INSERT INTO app_starts (ts) VALUES (?)", (time.time(),))
        db.get_db().commit()
    if not app.config["TESTING"]:
        heartbeat.start(app)

    app.register_blueprint(auth.bp)
    app.register_blueprint(views.bp)
    app.register_blueprint(dashboard.bp)
    app.register_blueprint(retrain_routes.bp)
    return app
