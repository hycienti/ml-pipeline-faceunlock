import json
import os
import platform
import time
from pathlib import Path

from flask import Blueprint, abort, current_app, jsonify, send_file

from app.auth import login_required
from app.db import get_db

bp = Blueprint("dashboard", __name__)


def uptime_summary(db, now: float | None = None, window_hours: int = 24, bucket_minutes: int = 5) -> dict:
    """Uptime in 5-minute slots.

    up   = a heartbeat arrived in the slot
    idle = no heartbeat, but the app had shut down cleanly and not yet been woken again
           (a scale-to-zero sleep): this still counts as up, because it answers on demand
    down = no heartbeat and no clean shutdown (a crash or an outage)
    """
    now = now or time.time()
    start = now - window_hours * 3600
    size = bucket_minutes * 60
    first = db.execute("SELECT MIN(ts) FROM heartbeats").fetchone()[0]
    beats = db.execute("SELECT ts FROM heartbeats WHERE ts >= ?", (start,)).fetchall()
    up_buckets = {int((row[0] - start) // size) for row in beats}
    stops = [r[0] for r in db.execute("SELECT ts FROM app_stops ORDER BY ts")]
    starts = [r[0] for r in db.execute("SELECT ts FROM app_starts ORDER BY ts")]

    def sleeping_at(t: float) -> bool:
        before = [s for s in stops if s <= t]
        return bool(before) and not any(before[-1] < st <= t for st in starts)

    buckets = []
    for i in range(window_hours * 3600 // size):
        t0 = start + i * size
        idle = False
        if first is None or t0 + size < first:
            state = None
        elif i in up_buckets:
            state = 1
        elif sleeping_at(t0 + size / 2):
            state, idle = 1, True
        else:
            state = 0
        buckets.append({"t": t0, "up": state, "idle": idle})

    measured = [b["up"] for b in buckets if b["up"] is not None]
    started = db.execute("SELECT MAX(ts) FROM app_starts").fetchone()[0]
    return {
        "running_since": started,
        "uptime_seconds": now - started if started else None,
        "uptime_pct": round(100 * sum(measured) / len(measured), 2) if measured else None,
        "buckets": buckets,
    }


def _read_json(path: str):
    file = Path(path)
    return json.loads(file.read_text()) if file.exists() else None


@bp.get("/api/uptime")
@login_required
def uptime():
    summary = uptime_summary(get_db())
    summary["deployment"] = {
        "platform": "Railway" if os.environ.get("RAILWAY_PROJECT_ID") else "Local",
        "python": platform.python_version(),
        "model_version": current_app.config["MODEL_VERSION"],
        "threshold": getattr(current_app.config["VERIFIER"], "threshold", None),
    }
    return jsonify(summary)


@bp.get("/api/metrics")
@login_required
def metrics():
    report = _read_json(current_app.config["METRICS_PATH"])
    return jsonify({"available": True, **report} if report else {"available": False})


@bp.get("/api/dataset-stats")
@login_required
def dataset_stats():
    stats = _read_json(current_app.config["DATASET_STATS_PATH"])
    return jsonify({"available": True, **stats} if stats else {"available": False})


@bp.get("/api/sample-faces.jpg")
@login_required
def sample_faces():
    path = Path(current_app.config["SAMPLE_FACES_PATH"])
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="image/jpeg")
