import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from src.preprocessing import scale_pixels


def pair_scores(
    embeddings: np.ndarray, rows_a: np.ndarray, rows_b: np.ndarray
) -> np.ndarray:
    return np.sum(embeddings[rows_a] * embeddings[rows_b], axis=1)


def evaluate_pairs(
    scores: np.ndarray, labels: np.ndarray, threshold: float | None = None
) -> dict:
    labels = np.asarray(labels)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    if threshold is None:
        threshold = float(thresholds[np.argmin(np.abs(fpr - (1 - tpr)))])

    accepted = scores >= threshold
    tp = int(np.sum(accepted & (labels == 1)))
    fp = int(np.sum(accepted & (labels == 0)))
    fn = int(np.sum(~accepted & (labels == 1)))
    tn = int(np.sum(~accepted & (labels == 0)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "accuracy": (tp + tn) / len(labels),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": float(roc_auc_score(labels, scores)),
        "far": fp / (fp + tn) if fp + tn else 0.0,
        "frr": fn / (fn + tp) if fn + tp else 0.0,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }


def embed_faces(model, faces: np.ndarray, batch_size: int = 256) -> np.ndarray:
    batches = [
        model.predict(scale_pixels(faces[i : i + batch_size]), verbose=0)
        for i in range(0, len(faces), batch_size)
    ]
    return np.concatenate(batches)


def threshold_for_far(scores: np.ndarray, labels: np.ndarray, far: float) -> float:
    """Loosest threshold that still lets in at most `far` of different-person pairs."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    allowed = np.where(fpr <= far)[0]
    threshold = float(thresholds[allowed[np.argmax(tpr[allowed])]])
    return threshold if np.isfinite(threshold) else float(np.max(scores)) + 1e-6
