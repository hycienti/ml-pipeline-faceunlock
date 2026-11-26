"""Lay the data out in the standard repo layout: data/train, data/val, data/test.

Copies the LFW photos of each person-level split (the exact split the notebook trains and
evaluates on) into  data/<split>/<person>/<photo>.jpg.  Needs LFW extracted in data/lfw
(see README) and is safe to re-run.

    python scripts/prepare_data.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import build_lfw_splits, export_splits  # noqa: E402


def main() -> None:
    counts = export_splits(build_lfw_splits(ROOT / "data/lfw"), ROOT / "data")
    for split, photos in counts.items():
        print(f"data/{split}: {photos} photos")


if __name__ == "__main__":
    main()
