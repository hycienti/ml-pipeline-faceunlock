import math

import numpy as np

from src.preprocessing import detect_and_crop, resize_face, scale_pixels


# Naming someone from a gallery of N people gives a stranger N chances to pass, so identification
# (/api/predict) demands more similarity than the 1:1 login, and the extra grows with N.
# PREDICT_THRESHOLD_MARGIN is the extra for a 15-person gallery. Measured on 300 random galleries of
# held-out people, this keeps a stranger from being wrongly named more than 4 to 16% of the time at
# every gallery size from 1 to 30 (a flat extra was too strict for small galleries and too loose for big ones).
PREDICT_THRESHOLD_MARGIN = 0.09
_MARGIN_REFERENCE_SIZE = 15


def predict_bar(threshold: float, n_registered: int, margin: float = PREDICT_THRESHOLD_MARGIN) -> float:
    """Similarity a photo needs before /api/predict names someone, given how many people are registered."""
    return threshold + margin * math.log2(max(n_registered, 1)) / math.log2(_MARGIN_REFERENCE_SIZE)


class FaceVerifier:
    def __init__(self, model, threshold: float):
        self.model = model
        self.threshold = threshold

    @classmethod
    def from_path(cls, model_path: str, threshold: float) -> "FaceVerifier":
        from tensorflow import keras

        return cls(keras.models.load_model(model_path), threshold)

    def embed_many(self, images: list[np.ndarray], batch_size: int = 64) -> list[np.ndarray | None]:
        """Embed many photos in batches; None for any photo without exactly one face."""
        results: list[np.ndarray | None] = [None] * len(images)
        ready = []
        for position, image in enumerate(images):
            face = detect_and_crop(image)
            if face is not None:
                ready.append((position, scale_pixels(resize_face(face))))
        if ready:
            embeddings = self.model.predict(
                np.stack([x for _, x in ready]), batch_size=batch_size, verbose=0
            )
            for (position, _), embedding in zip(ready, embeddings):
                results[position] = embedding
        return results

    def embed(self, image: np.ndarray) -> np.ndarray | None:
        return self.embed_many([image])[0]

    def enroll(self, images: list[np.ndarray]) -> np.ndarray:
        embeddings = []
        for i, image in enumerate(images, start=1):
            embedding = self.embed(image)
            if embedding is None:
                raise ValueError(f"photo {i}: expected exactly one face")
            embeddings.append(embedding)
        mean = np.mean(embeddings, axis=0)
        return mean / np.linalg.norm(mean)

    def compare(self, embedding: np.ndarray, stored: np.ndarray) -> tuple[bool, float]:
        """Cosine similarity of a fresh embedding against a stored template."""
        score = float(np.dot(embedding, stored))
        return score >= self.threshold, score

    def verify(self, image: np.ndarray, stored: np.ndarray) -> tuple[bool, float]:
        embedding = self.embed(image)
        if embedding is None:
            return False, -1.0
        return self.compare(embedding, stored)
