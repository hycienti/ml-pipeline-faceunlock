import threading
from pathlib import Path

import numpy as np
from flask import Blueprint, abort, current_app, jsonify, send_file, session
from PIL import Image

from app.auth import login_required
from app.db import get_db
from src.insights import embedding_map, login_margin, nearest_lookalikes, quality_score
from src.replay import load_replay

bp = Blueprint("insights", __name__)

SAMPLE_PEOPLE = 400
_build_lock = threading.Lock()


def _dataset(app):
    """One embedded photo per dataset person for the live model, cached in memory and on disk.

    Returns (embeddings, names, paths) or None when the replay set is not available.
    """
    cache = app.config.setdefault("INSIGHT_CACHE", {})
    version = app.config["MODEL_VERSION"]
    with _build_lock:
        if version in cache:
            return cache[version]
        zip_path = Path(app.config["REPLAY_ZIP"])
        if not zip_path.exists():
            return None

        store = Path(app.config["INSIGHT_DIR"]) / f"dataset_v{version}.npz"
        if store.exists():
            saved = np.load(store, allow_pickle=False)
            cache[version] = (saved["embeddings"], saved["names"].tolist(), saved["paths"].tolist())
            return cache[version]

        replay = load_replay(str(zip_path), app.config["REPLAY_DIR"])
        people = {**replay["train"], **replay["val"]}
        names = sorted(people)
        np.random.default_rng(0).shuffle(names)
        verifier = app.config["VERIFIER"]
        candidates = names[:SAMPLE_PEOPLE]
        photos = [people[name][0] for name in candidates]
        images = [np.asarray(Image.open(photo).convert("RGB")) for photo in photos]
        embeddings, kept, paths = [], [], []
        for name, photo, embedding in zip(candidates, photos, verifier.embed_many(images)):
            if embedding is not None:  # skip photos the detector cannot use
                embeddings.append(embedding)
                kept.append(name.replace("_", " "))
                paths.append(photo)
        if not embeddings:
            return None
        arrays = (np.stack(embeddings).astype(np.float32), kept, paths)
        store.parent.mkdir(parents=True, exist_ok=True)
        np.savez(store, embeddings=arrays[0], names=np.array(kept), paths=np.array(paths))
        cache[version] = arrays
        return arrays


def _own_photos(username: str) -> list[Path]:
    return sorted((Path(current_app.config["PHOTO_DIR"]) / username.lower()).glob("*.jpg"))


@bp.get("/api/insights")
@login_required
def insights():
    app = current_app._get_current_object()
    db = get_db()
    username = session["user"]
    user = db.execute("SELECT embedding FROM users WHERE username = ?", (username,)).fetchone()
    template = np.frombuffer(user["embedding"], dtype=np.float32)

    photos = []
    for path in _own_photos(username):
        report = quality_score(np.asarray(Image.open(path).convert("RGB")))
        photos.append({**report, "photo": path.name})
    quality = {
        "photos": photos,
        "overall": round(float(np.mean([p["overall"] for p in photos])), 3) if photos else None,
    }

    dataset = _dataset(app)
    others = [
        np.frombuffer(row["embedding"], dtype=np.float32)
        for row in db.execute("SELECT embedding FROM users WHERE username != ?", (username,))
    ]
    if dataset is None:
        map_data, lookalikes = {"available": False}, []
    else:
        embeddings, names, _ = dataset
        map_data = {
            "available": True,
            **embedding_map(embeddings, template, np.stack(others) if others else None),
        }
        lookalikes = [
            {"name": m["name"], "similarity": m["similarity"], "photo_id": m["index"]}
            for m in nearest_lookalikes(template, embeddings, names, k=3)
        ]

    rows = [
        (row["success"], row["score"])
        for row in db.execute(
            "SELECT success, score FROM login_events WHERE username = ? ORDER BY id", (username,)
        )
    ]
    threshold = getattr(app.config["VERIFIER"], "threshold", 0.0)
    return jsonify(
        quality=quality, map=map_data, lookalikes=lookalikes, margin=login_margin(rows, threshold)
    )


@bp.get("/api/insights/photo/<int:photo_id>")
@login_required
def dataset_photo(photo_id: int):
    dataset = _dataset(current_app._get_current_object())
    if dataset is None or not 0 <= photo_id < len(dataset[2]):
        abort(404)
    return send_file(dataset[2][photo_id], mimetype="image/jpeg")


@bp.get("/api/insights/my-photo/<int:index>")
@login_required
def my_photo(index: int):
    photos = _own_photos(session["user"])
    if not 0 <= index < len(photos):
        abort(404)
    return send_file(photos[index], mimetype="image/jpeg")
