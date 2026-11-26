import numpy as np
from flask import Blueprint, current_app, jsonify, request

from app.auth import NO_FACE, _message, _read_image, login_required
from app.db import get_db
from src.prediction import predict_bar

bp = Blueprint("predict", __name__)

TOP_K = 3


@bp.post("/api/predict")
@login_required
def predict():
    """Who is in this photo? Compares one image with every registered person.

    Returns the best match when it clears the (stricter) predict bar, otherwise "unknown" or a
    "possible" match, plus the closest registered people with their scores. Not a login attempt.
    """
    photo = request.files.get("photo")
    image = _read_image(photo) if photo else None
    if image is None:
        return _message(False, "Upload an image to predict.", 400)

    verifier = current_app.config["VERIFIER"]
    embedding = verifier.embed(image)
    if embedding is None:
        return _message(False, NO_FACE, 422)

    # The bar is stricter than the 1:1 login bar and grows with the number of registered people
    # (see predict_bar). A face between the two bars is reported as "possible".
    login_bar = float(verifier.threshold)
    users = get_db().execute("SELECT username, embedding FROM users").fetchall()
    bar = predict_bar(login_bar, len(users), current_app.config["PREDICT_THRESHOLD_MARGIN"])
    scored = []
    for user in users:
        stored = np.frombuffer(user["embedding"], dtype=np.float32)
        _, score = verifier.compare(embedding, stored)
        scored.append({"username": user["username"], "score": round(score, 3), "match": score >= bar})
    scored.sort(key=lambda row: -row["score"])

    best = scored[0] if scored else None
    possible = best["username"] if best and not best["match"] and best["score"] >= login_bar else None
    return jsonify(
        ok=True,
        prediction=best["username"] if best and best["match"] else None,
        match=bool(best and best["match"]),
        possible=possible,
        score=best["score"] if best else None,
        threshold=round(bar, 3),
        login_threshold=round(login_bar, 3),
        candidates=scored[:TOP_K],
    )
