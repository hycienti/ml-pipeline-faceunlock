import time

import numpy as np
from PIL import Image

from src.evaluation import evaluate_pairs


def evaluate_live(verifier, replay: dict) -> dict:
    """Score the live verifier on the held-out validation pairs, exactly as a real login would.

    Photos the live pipeline cannot read (no single face) make their pairs unusable; they are
    skipped and counted, which is itself a production signal.
    """
    started = time.time()
    pairs = replay["val_pairs"]
    paths = sorted({p for a, b, _ in pairs for p in (a, b)})
    embeddings = dict(
        zip(paths, verifier.embed_many([np.asarray(Image.open(p).convert("RGB")) for p in paths]))
    )

    scores, labels = [], []
    for a, b, label in pairs:
        if embeddings[a] is not None and embeddings[b] is not None:
            scores.append(float(np.dot(embeddings[a], embeddings[b])))
            labels.append(label)
    if len(set(labels)) < 2:
        raise ValueError("no usable validation pairs: the live pipeline found no faces to compare")

    metrics = evaluate_pairs(np.array(scores), np.array(labels), threshold=verifier.threshold)
    metrics.pop("fpr")
    metrics.pop("tpr")
    return {
        "metrics": metrics,
        "threshold": float(verifier.threshold),
        "pairs_evaluated": len(scores),
        "pairs_skipped_no_face": len(pairs) - len(scores),
        "seconds": round(time.time() - started, 2),
    }
