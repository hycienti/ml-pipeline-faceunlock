"""Add the login threshold to a model's metrics report.

The threshold is the loosest one that lets in at most FAR (default 2%) of different-person
pairs, measured on the original validation pairs from models/replay_faces.zip.
Run it after every training run that overwrites models/metrics_v1.json:

    python scripts/set_login_threshold.py [far]
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tensorflow import keras  # noqa: E402

from src.operating_point import DEFAULT_FAR, login_threshold  # noqa: E402
from src.replay import load_replay  # noqa: E402


def main(far: float = DEFAULT_FAR) -> None:
    metrics_path = ROOT / "models" / "metrics_v1.json"
    report = json.loads(metrics_path.read_text())
    model = keras.models.load_model(ROOT / "models" / "faceunlock_v1.keras")
    with tempfile.TemporaryDirectory() as tmp:
        replay = load_replay(str(ROOT / "models" / "replay_faces.zip"), tmp)
        point = login_threshold(model, replay, far)
    report["login_threshold"] = point["threshold"]
    report["login_operating_point"] = point
    metrics_path.write_text(json.dumps(report, indent=2))
    print(f"login threshold {point['threshold']:.3f} -> on validation pairs: "
          f"strangers accepted {point['far']:.1%}, own face rejected {point['frr']:.1%}")


if __name__ == "__main__":
    main(*(float(a) for a in sys.argv[1:]))
