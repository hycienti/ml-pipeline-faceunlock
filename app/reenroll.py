from pathlib import Path

import numpy as np
from PIL import Image


def reenroll_users(users, photo_dir: str, verifier) -> list[tuple[bytes, int]]:
    """Recompute every user's face template with `verifier` from their saved enrollment photos.

    Returns (embedding_bytes, user_id) pairs, or raises ValueError naming the first user who
    cannot be re-enrolled, so callers can apply all changes or none.
    """
    updated = []
    for user in users:
        photos = sorted((Path(photo_dir) / user["username"].lower()).glob("*.jpg"))
        try:
            if not photos:
                raise ValueError("no enrollment photos found")
            images = [np.asarray(Image.open(photo).convert("RGB")) for photo in photos]
            embedding = verifier.enroll(images)
        except ValueError as error:
            raise ValueError(
                f"Could not re-enroll {user['username']} with the new model: {error}"
            ) from None
        updated.append((embedding.astype(np.float32).tobytes(), user["id"]))
    return updated
