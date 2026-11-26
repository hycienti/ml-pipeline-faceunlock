import random

import cv2
import numpy as np
from PIL import Image

_DETECTOR = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_and_crop(image: np.ndarray, margin: float = 0.2) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = _DETECTOR.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) != 1:
        return None

    x, y, w, h = faces[0]
    pad_x, pad_y = int(w * margin), int(h * margin)
    x0, y0 = max(x - pad_x, 0), max(y - pad_y, 0)
    x1 = min(x + w + pad_x, image.shape[1])
    y1 = min(y + h + pad_y, image.shape[0])
    return image[y0:y1, x0:x1]


def resize_face(face: np.ndarray, size: int = 160) -> np.ndarray:
    image = Image.fromarray(face).convert("RGB").resize(
        (size, size), Image.Resampling.LANCZOS
    )
    return np.asarray(image, dtype=np.uint8)


def scale_pixels(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) / 127.5 - 1.0


def preprocess(face: np.ndarray, size: int = 160) -> np.ndarray:
    return scale_pixels(resize_face(face, size))


def center_crop(image: np.ndarray, fraction: float = 0.65) -> np.ndarray:
    h, w = image.shape[:2]
    ch, cw = int(h * fraction), int(w * fraction)
    top, left = (h - ch) // 2, (w - cw) // 2
    return image[top : top + ch, left : left + cw]


def build_pairs(
    index: dict[str, list[str]], n_pairs: int, seed: int
) -> list[tuple[str, str, int]]:
    rng = random.Random(seed)
    people = sorted(p for p, photos in index.items() if photos)
    multi = [p for p in people if len(index[p]) >= 2]
    if not multi or len(people) < 2:
        raise ValueError("need 2+ people and at least one person with 2+ photos")

    pairs = []
    for i in range(n_pairs):
        if i % 2 == 0:
            person = rng.choice(multi)
            a, b = rng.sample(index[person], 2)
            pairs.append((a, b, 1))
        else:
            p1, p2 = rng.sample(people, 2)
            pairs.append((rng.choice(index[p1]), rng.choice(index[p2]), 0))
    rng.shuffle(pairs)
    return pairs


def load_training_face(path: str) -> np.ndarray:
    """Training-time recipe: detect one face, else fall back to a centre crop; 160x160 uint8."""
    image = np.asarray(Image.open(path).convert("RGB"))
    face = detect_and_crop(image)
    if face is None:
        face = center_crop(image)
    return resize_face(face)
