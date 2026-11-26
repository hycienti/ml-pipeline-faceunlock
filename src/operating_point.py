import numpy as np

from src.evaluation import embed_faces, evaluate_pairs, pair_scores, threshold_for_far
from src.preprocessing import load_training_face

DEFAULT_FAR = 0.02


def login_threshold(model, replay: dict, far: float = DEFAULT_FAR) -> dict:
    """Pick the login threshold on the original validation pairs for a target false-accept rate."""
    pairs = replay["val_pairs"]
    paths = sorted({p for a, b, _ in pairs for p in (a, b)})
    row = {p: i for i, p in enumerate(paths)}
    embeddings = embed_faces(model, np.stack([load_training_face(p) for p in paths]))
    scores = pair_scores(
        embeddings,
        np.array([row[a] for a, _, _ in pairs]),
        np.array([row[b] for _, b, _ in pairs]),
    )
    labels = np.array([label for _, _, label in pairs])
    threshold = threshold_for_far(scores, labels, far)
    at_threshold = evaluate_pairs(scores, labels, threshold=threshold)
    return {
        "threshold": threshold,
        "far_target": far,
        "far": at_threshold["far"],
        "frr": at_threshold["frr"],
    }
