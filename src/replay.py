import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path


def _extract_once(zip_path: str, dest: Path) -> None:
    """Unzip into a private staging folder, then rename it into place in one atomic step.

    Several containers share one volume and all start at once. Whoever renames first wins;
    the others discard their copy and use it. Nobody ever sees a half-extracted folder.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".replay-", dir=dest.parent))
    try:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(staging)
        (staging / ".extracted").touch()
        if dest.exists() and not (dest / ".extracted").exists():
            shutil.rmtree(dest, ignore_errors=True)  # left over from an interrupted extraction
        try:
            os.rename(staging, dest)
            staging = None
        except OSError:
            pass  # another process finished first
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


def load_replay(zip_path: str, dest_dir: str) -> dict:
    """Unzip the replay set once and index it.

    Returns {"train": {person: [paths]}, "val": {person: [paths]},
             "val_pairs": [(path_a, path_b, label)]}.
    """
    dest = Path(dest_dir)
    if not (dest / ".extracted").exists():
        _extract_once(zip_path, dest)

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
