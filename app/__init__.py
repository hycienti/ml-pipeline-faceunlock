import json
import os
import time
from pathlib import Path

from flask import Flask

from app import auth, dashboard, db, heartbeat, views

ROOT = Path(__file__).resolve().parent.parent


def load_verifier(app: Flask):
    from src.prediction import FaceVerifier

    metrics = json.loads(Path(app.config["METRICS_PATH"]).read_text())
    return FaceVerifier.from_path(app.config["MODEL_PATH"], metrics["threshold"])


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__, instance_path=str(ROOT / "instance"))
    app.config.update(
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev-only-change-me"),
        DB_PATH=str(Path(app.instance_path) / "faceunlock.db"),
        PHOTO_DIR=str(Path(app.instance_path) / "photos"),
        MODEL_PATH=str(ROOT / "models" / "faceunlock_v1.keras"),
        METRICS_PATH=str(ROOT / "models" / "metrics_v1.json"),
        DATASET_STATS_PATH=str(ROOT / "models" / "dataset_stats.json"),
        HEARTBEAT_SECONDS=60,
        MODEL_VERSION=1,
        MAX_FAILED_LOGINS=5,
        LOCKOUT_SECONDS=300,
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,
    )
    if test_config:
        app.config.update(test_config)

    Path(app.config["DB_PATH"]).parent.mkdir(parents=True, exist_ok=True)
    Path(app.config["PHOTO_DIR"]).mkdir(parents=True, exist_ok=True)
    if "VERIFIER" not in app.config:
        app.config["VERIFIER"] = load_verifier(app)

    app.teardown_appcontext(db.close_db)
    with app.app_context():
        db.init_db()
        db.get_db().execute("INSERT INTO app_starts (ts) VALUES (?)", (time.time(),))
        db.get_db().commit()
    if not app.config["TESTING"]:
        heartbeat.start(app)

    app.register_blueprint(auth.bp)
    app.register_blueprint(views.bp)
    app.register_blueprint(dashboard.bp)
    return app
