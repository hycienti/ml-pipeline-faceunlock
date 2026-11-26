import io
import json
import zipfile

import numpy as np
from PIL import Image

from src.replay import load_replay


def jpeg_bytes(seed):
    buffer = io.BytesIO()
    pixels = np.random.default_rng(seed).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(buffer, "JPEG")
    return buffer.getvalue()


def make_replay_zip(path, train_people=12, val_people=4):
    pairs = []
    with zipfile.ZipFile(path, "w") as archive:
        for i in range(train_people):
            for j in range(2):
                archive.writestr(f"train/t{i}/t{i}_{j}.jpg", jpeg_bytes(i * 10 + j))
        for i in range(val_people):
            for j in range(2):
                archive.writestr(f"val/v{i}/v{i}_{j}.jpg", jpeg_bytes(500 + i * 10 + j))
            pairs.append([f"val/v{i}/v{i}_0.jpg", f"val/v{i}/v{i}_1.jpg", 1])
            pairs.append([f"val/v{i}/v{i}_0.jpg", f"val/v{(i + 1) % val_people}/v{(i + 1) % val_people}_1.jpg", 0])
        archive.writestr("val_pairs.json", json.dumps(pairs))
    return path


def test_load_replay_returns_indexes_and_absolute_pairs(tmp_path):
    zip_path = make_replay_zip(tmp_path / "replay.zip")
    replay = load_replay(str(zip_path), str(tmp_path / "out"))
    assert len(replay["train"]) == 12 and len(replay["val"]) == 4
    assert all(len(photos) == 2 for photos in replay["train"].values())
    first_a, first_b, label = replay["val_pairs"][0]
    assert first_a.startswith(str(tmp_path / "out")) and first_b.endswith(".jpg") and label == 1
    assert (tmp_path / "out" / "val_pairs.json").exists()


def test_load_replay_extracts_once(tmp_path):
    zip_path = make_replay_zip(tmp_path / "replay.zip")
    load_replay(str(zip_path), str(tmp_path / "out"))
    marker = next((tmp_path / "out").glob("train/t0/*.jpg"))
    marker.write_bytes(b"changed")
    load_replay(str(zip_path), str(tmp_path / "out"))  # second call must not re-extract
    assert marker.read_bytes() == b"changed"
