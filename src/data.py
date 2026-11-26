import csv
import random
from pathlib import Path


def build_lfw_splits(
    lfw_root: str,
    max_photos: int = 20,
    test_fraction: float = 0.2,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, dict[str, list[str]]]:
    """Person-level train/val/test split of LFW: {split: {person: [photo paths]}}.

    People with fewer than 2 photos are skipped, LFW's official test people are removed
    (they stay untouched for the official benchmark), and each person is capped at
    `max_photos` randomly chosen photos.
    """
    root = Path(lfw_root)
    faces_dir = root / "lfw-deepfunneled" / "lfw-deepfunneled"
    rng = random.Random(seed)
    with open(root / "peopleDevTest.csv", newline="") as handle:
        official_test_people = {row["name"] for row in csv.DictReader(handle)}

    index = {}
    for person_dir in sorted(faces_dir.iterdir()):
        if person_dir.name in official_test_people:
            continue
        photos = sorted(person_dir.glob("*.jpg"))
        if len(photos) >= 2:
            keep = rng.sample(photos, min(len(photos), max_photos))
            index[person_dir.name] = [str(p) for p in sorted(keep)]

    names = sorted(index)
    rng.shuffle(names)
    n_test = int(len(names) * test_fraction)
    n_val = int(len(names) * val_fraction)
    return {
        "test": {n: index[n] for n in names[:n_test]},
        "val": {n: index[n] for n in names[n_test : n_test + n_val]},
        "train": {n: index[n] for n in names[n_test + n_val :]},
    }
