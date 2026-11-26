import atexit
import functools
import json
import os
import secrets
import threading
import time
from pathlib import Path

from flask import Flask

from src.prediction import PREDICT_THRESHOLD_MARGIN
from app import auth, dashboard, db, heartbeat, insights, predict, retrain_routes, views

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
    threshold = metrics.get("login_threshold", metrics["threshold"])
    return app.config["VERIFIER_FACTORY"](app.config["MODEL_PATH"], threshold)


def _persistent_secret_key(folder: Path) -> str:
    """A random session key stored once in the instance folder, so restarts and containers sharing
    the folder agree on it. Exclusive creation means two first starts cannot write different keys."""
    folder.mkdir(parents=True, exist_ok=True)
    key_file = folder / "secret_key"
    try:
        descriptor = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return key_file.read_text().strip()
    with os.fdopen(descriptor, "w") as handle:
        handle.write(secrets.token_hex(32))
    return key_file.read_text().strip()


def create_app(test_config: dict | None = None) -> Flask:
    instance = os.environ.get("FACEUNLOCK_INSTANCE", str(ROOT / "instance"))  # persistent storage in production
    app = Flask(__name__, instance_path=instance)
    app.config.update(
        SECRET_KEY=os.environ.get("SECRET_KEY"),  # else a random key kept in the instance folder
        DB_PATH=str(Path(app.instance_path) / "faceunlock.db"),
        PHOTO_DIR=str(Path(app.instance_path) / "photos"),
        MODEL_PATH=str(ROOT / "models" / "faceunlock_v1.keras"),
        METRICS_PATH=str(ROOT / "models" / "metrics_v1.json"),
        DATASET_STATS_PATH=str(ROOT / "models" / "dataset_stats.json"),
        SAMPLE_FACES_PATH=str(ROOT / "models" / "sample_faces.jpg"),
        REPLAY_ZIP=str(ROOT / "models" / "replay_faces.zip"),
        REPLAY_DIR=str(Path(app.instance_path) / "replay"),
        INSIGHT_DIR=str(Path(app.instance_path) / "insights"),
        RETRAIN_DIR=str(Path(app.instance_path) / "retrain"),
        ACTIVE_MODELS_DIR=str(Path(app.instance_path) / "models"),
        VERIFIER_FACTORY=verifier_from_files,
        HEARTBEAT_SECONDS=60,
        RETRAIN_RECOMMEND_PEOPLE=int(os.environ.get("RETRAIN_RECOMMEND_PEOPLE", 10)),
        AUTO_RETRAIN_PEOPLE=int(os.environ.get("AUTO_RETRAIN_PEOPLE", 0)),  # 0 = never retrain automatically
        MODEL_VERSION=1,
        # extra similarity /api/predict demands on top of the login bar for a 15-person gallery (see predict_bar)
        PREDICT_THRESHOLD_MARGIN=float(os.environ.get("PREDICT_THRESHOLD_MARGIN", PREDICT_THRESHOLD_MARGIN)),
        MAX_FAILED_LOGINS=5,
        LOCKOUT_SECONDS=300,
        MAX_CONTENT_LENGTH=512 * 1024 * 1024,  # retrain zips can be large
    )
    if test_config:
        app.config.update(test_config)

    Path(app.config["DB_PATH"]).parent.mkdir(parents=True, exist_ok=True)
    if not app.config.get("SECRET_KEY"):
        app.config["SECRET_KEY"] = _persistent_secret_key(Path(app.instance_path))
    Path(app.config["PHOTO_DIR"]).mkdir(parents=True, exist_ok=True)
    if "RETRAIN_FN" not in app.config:
        replay = Path(app.config["REPLAY_ZIP"])
        app.config["RETRAIN_FN"] = functools.partial(
            run_retrain_job,
            replay_zip=str(replay) if replay.exists() else None,
            replay_dir=app.config["REPLAY_DIR"],
        )
    app.config.setdefault("DATASETS_DIR", str(Path(app.config["RETRAIN_DIR"]).parent / "datasets"))
    Path(app.config["DATASETS_DIR"]).mkdir(parents=True, exist_ok=True)
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
        atexit.register(heartbeat.record_stop, app)

    app.register_blueprint(auth.bp)
    app.register_blueprint(views.bp)
    app.register_blueprint(dashboard.bp)
    app.register_blueprint(insights.bp)
    app.register_blueprint(predict.bp)
    app.register_blueprint(retrain_routes.bp)
    if not app.config["TESTING"] and Path(app.config["REPLAY_ZIP"]).exists():
        threading.Thread(target=insights._dataset, args=(app,), daemon=True, name="insight-warmup").start()
    return app
