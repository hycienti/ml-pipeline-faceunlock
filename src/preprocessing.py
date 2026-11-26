import cv2
import numpy as np

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
