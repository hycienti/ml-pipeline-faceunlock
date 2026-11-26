import cv2
import numpy as np

from src.preprocessing import detect_face_box


def _clip01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def quality_score(image: np.ndarray) -> dict:
    """How good a registration photo is: brightness, sharpness, face size, plus retake hints."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    brightness = float(gray.mean())
    height, width = gray.shape
    small = cv2.resize(gray, (320, max(int(height * 320 / width), 1)))  # compare photos at one scale
    sharpness = float(cv2.Laplacian(small, cv2.CV_64F).var())
    box = detect_face_box(image)
    face_ratio = None if box is None else box[2] / min(height, width)

    scores = {
        "brightness": _clip01(1 - abs(brightness - 120) / 120),
        "sharpness": _clip01(sharpness / 100),
    }
    hints = []
    if brightness < 70:
        hints.append("Too dark: face a window or a lamp.")
    elif brightness > 190:
        hints.append("Too bright: avoid glare and strong backlight.")
    if sharpness < 40:
        hints.append("Blurry: hold still and wipe the camera lens.")
    if face_ratio is None:
        hints.append("No single face found: centre your face, and make sure you are alone in the frame.")
    else:
        scores["face_size"] = _clip01(1 - abs(face_ratio - 0.45) / 0.45)
        if face_ratio < 0.25:
            hints.append("Too far away: move closer to the camera.")
        elif face_ratio > 0.8:
            hints.append("Too close: move back a little.")
    return {
        "brightness": round(brightness, 1),
        "sharpness": round(sharpness, 1),
        "face_ratio": None if face_ratio is None else round(face_ratio, 3),
        "scores": {k: round(v, 3) for k, v in scores.items()},
        "overall": round(float(np.mean(list(scores.values()))), 3),
        "hints": hints,
    }


def embedding_map(dataset: np.ndarray, user: np.ndarray, others: np.ndarray | None = None) -> dict:
    """Project embeddings to 2D with PCA fitted on the dataset faces."""
    center = dataset.mean(axis=0)
    _, singular, axes = np.linalg.svd(dataset - center, full_matrices=False)
    axes = axes[:2]
    if len(axes) < 2:  # degenerate data: pad so we always return 2D points
        axes = np.vstack([axes, np.zeros((2 - len(axes), dataset.shape[1]))])

    def project(matrix: np.ndarray) -> list[list[float]]:
        return ((np.atleast_2d(matrix) - center) @ axes.T).round(4).tolist()

    variance = singular**2 / np.sum(singular**2)
    return {
        "dataset": project(dataset),
        "user": project(user)[0],
        "others": project(others) if others is not None and len(others) else [],
        "explained": [round(float(v), 3) for v in variance[:2]],
    }


def nearest_lookalikes(user: np.ndarray, embeddings: np.ndarray, names: list[str], k: int = 3) -> list[dict]:
    similarities = embeddings @ user
    order = np.argsort(-similarities)[:k]
    return [
        {"index": int(i), "name": names[i], "similarity": round(float(similarities[i]), 3)}
        for i in order
    ]


def login_margin(rows: list[tuple[int, float | None]], threshold: float) -> dict:
    """Summarise login attempts: how far above the threshold the successful ones land."""
    successes = [score for ok, score in rows if ok and score is not None]
    mean_success = float(np.mean(successes)) if successes else None
    return {
        "attempts": len(rows),
        "successes": sum(1 for ok, _ in rows if ok),
        "threshold": threshold,
        "mean_success_score": None if mean_success is None else round(mean_success, 3),
        "margin": None if mean_success is None else round(mean_success - threshold, 3),
        "recent": [{"ok": bool(ok), "score": score} for ok, score in rows[-20:]],
    }


def dataset_stories(stats: dict) -> list[dict]:
    """Plain-language interpretations of the dataset features, written from the real numbers."""
    stories = []
    lfw = stats.get("lfw")
    if lfw:
        single_pct = 100 * lfw["single_photo_people"] / lfw["people"]
        usable_pct = 100 * lfw["usable_people"] / lfw["people"]
        stories.append({
            "feature": "Photos per person",
            "story": (
                f"{lfw['single_photo_people']:,} of {lfw['people']:,} people ({single_pct:.1f}%) have a single photo, "
                f"while {lfw['top_person'].replace('_', ' ')} alone has {lfw['top_count']}. Verification needs at least "
                f"two photos of a person, so only {lfw['usable_people']:,} people ({usable_pct:.0f}%) are usable. "
                f"We cap every person at {stats.get('photo_cap', 20)} photos so a few celebrities do not dominate "
                "what the model learns."
            ),
        })
    kaggle = stats.get("kaggle", {})
    width = kaggle.get("width_px")
    if width:
        stories.append({
            "feature": "Image size",
            "story": (
                f"The Kaggle faces run from {width['min']:,}px to {width['max']:,}px wide (median {width['median']:,}px). "
                "A CNN needs one fixed input size, so every face is detected, cropped and resized to 160x160. "
                "Without that, the biggest photos would carry far more detail than the smallest."
            ),
        })
    light = kaggle.get("brightness_stats")
    if light:
        stories.append({
            "feature": "Lighting",
            "story": (
                f"Average brightness is {light['mean']:.0f} out of 255 but ranges from {light['min']:.0f} (very dark) to "
                f"{light['max']:.0f} (washed out), with a spread of {light['std']:.0f}. A model trained on evenly lit "
                "photos would fail on a dim webcam, which is why training applies brightness and contrast augmentation."
            ),
        })
    if "detector_success_rate" in stats:
        rate = 100 * stats["detector_success_rate"]
        stories.append({
            "feature": "Faces per photo",
            "story": (
                f"The face detector finds exactly one face in {rate:.1f}% of training photos; the rest are group shots "
                "or unusual angles. The app refuses such photos at registration and login ('no single face'), and "
                "training falls back to a centre crop for them."
            ),
        })
    modes = kaggle.get("colour_modes")
    if modes and kaggle.get("sampled"):
        odd = kaggle["sampled"] - modes.get("RGB", 0)
        stories.append({
            "feature": "Colour format",
            "story": (
                f"{odd} of {kaggle['sampled']} sampled Kaggle images are not plain RGB (grayscale, PNGs with "
                "transparency, palette images). Every image is converted to 3-channel RGB first so the network always "
                "sees the same kind of input."
            ),
        })
    return stories
