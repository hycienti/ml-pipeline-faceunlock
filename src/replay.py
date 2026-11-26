import json
import shutil
import zipfile
from pathlib import Path


def load_replay(zip_path: str, dest_dir: str) -> dict:
    """Unzip the replay set once and index it.

    Returns {"train": {person: [paths]}, "val": {person: [paths]},
             "val_pairs": [(path_a, path_b, label)]}.
    """
    dest = Path(dest_dir)
    marker = dest / ".extracted"
    if not marker.exists():
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(dest)
        marker.touch()

    def index(split: str) -> dict[str, list[str]]:
        return {
            person.name: sorted(str(p) for p in person.glob("*.jpg"))
            for person in sorted((dest / split).iterdir())
        }

    pairs = [
        (str(dest / a), str(dest / b), label)
        for a, b, label in json.loads((dest / "val_pairs.json").read_text())
    ]
    return {"train": index("train"), "val": index("val"), "val_pairs": pairs}
