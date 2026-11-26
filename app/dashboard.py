import json
import os
import platform
import time
from pathlib import Path

from flask import Blueprint, abort, current_app, jsonify, send_file

from app.auth import login_required
from app.db import get_db
from src.insights import dataset_stories
from src.live_evaluation import evaluate_live
from src.replay import load_replay

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
    if not stats:
        return jsonify({"available": False})
    return jsonify({"available": True, **stats, "stories": dataset_stories(stats)})


@bp.get("/api/sample-faces.jpg")
@login_required
def sample_faces():
    path = Path(current_app.config["SAMPLE_FACES_PATH"])
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="image/jpeg")


def _evaluation_dict(row) -> dict:
    return {"id": row["id"], "model_version": row["model_version"], "ts": row["ts"], **json.loads(row["result"])}


def _evaluation_state(db) -> dict:
    rows = db.execute("SELECT * FROM live_evaluations ORDER BY id DESC LIMIT 10").fetchall()
    since = time.time() - 24 * 3600
    attempts, passes = db.execute(
        "SELECT COUNT(*), COALESCE(SUM(success), 0) FROM login_events WHERE ts > ?", (since,)
    ).fetchone()
    return {
        "latest": _evaluation_dict(rows[0]) if rows else None,
        "history": [_evaluation_dict(r) for r in rows],
        "traffic": {
            "window_hours": 24,
            "attempts": attempts,
            "passes": passes,
            "pass_rate": round(passes / attempts, 3) if attempts else None,
        },
    }


@bp.get("/api/evaluate")
@login_required
def evaluation():
    return jsonify(_evaluation_state(get_db()))


@bp.post("/api/evaluate")
@login_required
def run_evaluation():
    """Evaluate the model that is serving requests right now and keep the result."""
    zip_path = Path(current_app.config["REPLAY_ZIP"])
    if not zip_path.exists():
        return jsonify(ok=False, message="The replay set (models/replay_faces.zip) is missing, so there are no validation pairs to evaluate on."), 503
    replay = load_replay(str(zip_path), current_app.config["REPLAY_DIR"])
    try:
        result = evaluate_live(current_app.config["VERIFIER"], replay)
    except ValueError as error:
        return jsonify(ok=False, message=str(error)), 422
    db = get_db()
    db.execute(
        "INSERT INTO live_evaluations (model_version, ts, result) VALUES (?, ?, ?)",
        (current_app.config["MODEL_VERSION"], time.time(), json.dumps(result)),
    )
    db.commit()
    return jsonify(ok=True, **_evaluation_state(db))
