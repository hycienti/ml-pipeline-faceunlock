import numpy as np
from flask import Blueprint, current_app, jsonify, request

from app.auth import NO_FACE, _message, _read_image, login_required
from app.db import get_db

bp = Blueprint("predict", __name__)

TOP_K = 3


@bp.post("/api/predict")
@login_required
def predict():
    """Who is in this photo? Compares one image with every registered person.

    Returns the best match when it clears the login threshold, otherwise "unknown",
    plus the closest registered people with their scores. Not a login attempt.
    """
    photo = request.files.get("photo")
    image = _read_image(photo) if photo else None
    if image is None:
        return _message(False, "Upload an image to predict.", 400)

    verifier = current_app.config["VERIFIER"]
    embedding = verifier.embed(image)
    if embedding is None:
        return _message(False, NO_FACE, 422)

    scored = []
    for user in get_db().execute("SELECT username, embedding FROM users"):
        stored = np.frombuffer(user["embedding"], dtype=np.float32)
        match, score = verifier.compare(embedding, stored)
        scored.append({"username": user["username"], "score": round(score, 3), "match": match})
    scored.sort(key=lambda row: -row["score"])

    best = scored[0] if scored else None
    return jsonify(
        ok=True,
        prediction=best["username"] if best and best["match"] else None,
        match=bool(best and best["match"]),
        score=best["score"] if best else None,
        threshold=round(float(verifier.threshold), 3),
        candidates=scored[:TOP_K],
    )
