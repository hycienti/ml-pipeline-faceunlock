"""Save models/faceunlock_v1.keras also as models/faceunlock_v1.h5 (the course asks for .pkl, .tf or .h5).

Run after every training run or manual model swap:
    python scripts/export_h5.py
"""
import sys
from pathlib import Path

import numpy as np
from tensorflow import keras

MODELS = Path(__file__).resolve().parent.parent / "models"


def main() -> None:
    model = keras.models.load_model(MODELS / "faceunlock_v1.keras")
    model.save(MODELS / "faceunlock_v1.h5")
    reloaded = keras.models.load_model(MODELS / "faceunlock_v1.h5")
    images = np.random.default_rng(0).uniform(-1, 1, (4, 160, 160, 3)).astype("float32")
    gap = float(np.abs(model.predict(images, verbose=0) - reloaded.predict(images, verbose=0)).max())
    size = (MODELS / "faceunlock_v1.h5").stat().st_size / 1e6
    print(f"wrote models/faceunlock_v1.h5 ({size:.1f} MB); largest difference from the .keras model: {gap:.2e}")
    if gap > 1e-4:
        sys.exit("the .h5 model does not match the .keras model")


if __name__ == "__main__":
    main()
