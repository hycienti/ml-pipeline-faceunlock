import numpy as np

from src.preprocessing import detect_and_crop, resize_face, scale_pixels


class FaceVerifier:
    def __init__(self, model, threshold: float):
        self.model = model
        self.threshold = threshold

    @classmethod
    def from_path(cls, model_path: str, threshold: float) -> "FaceVerifier":
        from tensorflow import keras

        return cls(keras.models.load_model(model_path), threshold)

    def embed(self, image: np.ndarray) -> np.ndarray | None:
        face = detect_and_crop(image)
        if face is None:
            return None
        x = scale_pixels(resize_face(face))[None]
        return self.model.predict(x, verbose=0)[0]

    def enroll(self, images: list[np.ndarray]) -> np.ndarray:
        embeddings = []
        for i, image in enumerate(images, start=1):
            embedding = self.embed(image)
            if embedding is None:
                raise ValueError(f"photo {i}: expected exactly one face")
            embeddings.append(embedding)
        mean = np.mean(embeddings, axis=0)
        return mean / np.linalg.norm(mean)

    def verify(self, image: np.ndarray, stored: np.ndarray) -> tuple[bool, float]:
        embedding = self.embed(image)
        if embedding is None:
            return False, -1.0
        score = float(np.dot(embedding, stored))
        return score >= self.threshold, score
