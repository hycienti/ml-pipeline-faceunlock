import re
import time
from functools import wraps
from pathlib import Path

import numpy as np
from flask import Blueprint, current_app, jsonify, redirect, request, session, url_for
from PIL import Image, UnidentifiedImageError

from app.db import get_db

bp = Blueprint("auth", __name__)

USERNAME_RE = re.compile(r"[A-Za-z0-9_.-]{3,30}")
REQUIRED_PHOTOS = 3
LOGIN_FAILED = "Login failed."
NO_FACE = "We could not find exactly one face in that photo. Face the camera in good light, alone in the frame."


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("views.login_page"))
        return view(*args, **kwargs)

    return wrapped


def _read_image(file_storage) -> np.ndarray | None:
    try:
        image = Image.open(file_storage.stream)
        image.load()
        return np.asarray(image.convert("RGB"))
    except (UnidentifiedImageError, OSError):
        return None


def _message(ok: bool, text: str, status: int, **extra):
    return jsonify(ok=ok, message=text, **extra), status


def _is_locked(db, username: str) -> bool:
    since = time.time() - current_app.config["LOCKOUT_SECONDS"]
    last_success = db.execute(
        "SELECT COALESCE(MAX(ts), 0) FROM login_events WHERE username = ? AND success = 1",
        (username,),
    ).fetchone()[0]
    failures = db.execute(
        "SELECT COUNT(*) FROM login_events WHERE username = ? AND success = 0 AND ts > ?",
        (username, max(since, last_success)),
    ).fetchone()[0]
    return failures >= current_app.config["MAX_FAILED_LOGINS"]


def _log_event(db, username: str, success: bool, score) -> None:
    db.execute(
        "INSERT INTO login_events (username, success, score, ts) VALUES (?, ?, ?, ?)",
        (username, int(success), score, time.time()),
    )
    db.commit()


@bp.post("/api/register")
def register():
    username = request.form.get("username", "").strip()
    if not USERNAME_RE.fullmatch(username):
        return _message(
            False, "Username must be 3-30 letters, digits, '.', '_' or '-'.", 400
        )
    files = request.files.getlist("photos")
    if len(files) != REQUIRED_PHOTOS:
        return _message(False, f"Please provide exactly {REQUIRED_PHOTOS} photos.", 400)
    images = [_read_image(f) for f in files]
    if any(image is None for image in images):
        return _message(False, "One of the files is not a valid image.", 400)

    db = get_db()
    if db.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone():
        return _message(False, "That username is taken.", 409)
    try:
        embedding = current_app.config["VERIFIER"].enroll(images)
    except ValueError as error:
        return _message(False, str(error), 400)

    try:
        db.execute(
            "INSERT INTO users (username, embedding, model_version, created_at)"
            " VALUES (?, ?, ?, ?)",
            (
                username,
                embedding.astype(np.float32).tobytes(),
                current_app.config["MODEL_VERSION"],
                time.time(),
            ),
        )
        db.commit()
    except Exception:
        return _message(False, "That username is taken.", 409)

    folder = Path(current_app.config["PHOTO_DIR"]) / username.lower()
    folder.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images, start=1):
        Image.fromarray(image).save(folder / f"{i}.jpg", quality=92)
    return _message(True, "Registered. You can log in now.", 201)


@bp.post("/api/login")
def login():
    username = request.form.get("username", "").strip()
    photo = request.files.get("photo")
    image = _read_image(photo) if photo else None
    if not username or image is None:
        return _message(False, "Enter your username and provide a photo.", 400)

    db = get_db()
    if _is_locked(db, username):
        return _message(False, "Too many failed attempts. Try again in a few minutes.", 429)

    verifier = current_app.config["VERIFIER"]
    embedding = verifier.embed(image)  # same work for known and unknown usernames
    if embedding is None:
        # Says nothing about who the user is, and a retake is not a failed attempt.
        return _message(False, NO_FACE, 422)

    user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    accepted, score = False, None
    if user is not None:
        stored = np.frombuffer(user["embedding"], dtype=np.float32)
        accepted, score = verifier.compare(embedding, stored)

    _log_event(db, username, accepted, score)
    if not accepted:
        return _message(False, LOGIN_FAILED, 401)

    session.clear()
    session["user"] = user["username"]
    return _message(True, "Welcome back.", 200, redirect=url_for("views.dashboard"))


@bp.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("views.login_page"))
