"""Build models/replay_faces.zip: the LFW photos the model was trained and validated on.

The deployed app unzips this during retraining so a retrain can mix the original people
with newly uploaded ones (and check that the old people are not forgotten).

Usage (from the repo root, with LFW extracted in data/lfw):
    python scripts/make_replay_zip.py
"""
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import build_lfw_splits  # noqa: E402
from src.preprocessing import build_pairs  # noqa: E402

SEED = 42
VAL_PAIRS = 1000


def main(lfw_root: str = str(ROOT / "data/lfw"), out: str = str(ROOT / "models/replay_faces.zip")):
    splits = build_lfw_splits(lfw_root, seed=SEED)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as archive:  # JPEGs do not compress further
        for split in ("train", "val"):
            for person, photos in splits[split].items():
                for photo in photos:
                    archive.write(photo, f"{split}/{person}/{Path(photo).name}")
        relative = lambda p: f"val/{Path(p).parent.name}/{Path(p).name}"
        pairs = build_pairs(splits["val"], VAL_PAIRS, SEED)
        archive.writestr(
            "val_pairs.json", json.dumps([[relative(a), relative(b), label] for a, b, label in pairs])
        )
    size = Path(out).stat().st_size / 1e6
    print(f"wrote {out}: {size:.1f} MB, "
          f"{sum(map(len, splits['train'].values()))} train and "
          f"{sum(map(len, splits['val'].values()))} validation photos")


if __name__ == "__main__":
    main(*sys.argv[1:])
