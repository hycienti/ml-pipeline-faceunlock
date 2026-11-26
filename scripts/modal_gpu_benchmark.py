"""Throwaway benchmark: does TensorFlow see a Modal GPU, and how fast is a real retrain on it?

It touches nothing in the deployed app. It uploads the model, the replay set and the demo retrain zip
to a short-lived Modal job on one T4 GPU, runs a light and a deep retrain, and prints the timings and
the old-versus-new AUC. Cost: a few cents of GPU time plus the one-off image build.

    source .venv/bin/activate
    modal run scripts/modal_gpu_benchmark.py

Needs data/demo_retrain/pins_20_people.zip (a zip of person folders) next to the repo.
"""
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
DEMO_ZIP = REPO / "data" / "demo_retrain" / "pins_20_people.zip"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libglib2.0-0", "libgl1")
    .pip_install_from_requirements(str(REPO / "requirements-app.txt"))
    .pip_install("tensorflow[and-cuda]")  # the CUDA libraries TensorFlow needs to use the GPU
    .env({"TF_CPP_MIN_LOG_LEVEL": "2"})
    .add_local_dir(str(REPO / "src"), "/root/src", ignore=["**/__pycache__"])
    .add_local_dir(str(REPO / "models"), "/root/models", ignore=["*.h5", "*.ipynb"])
    .add_local_file(str(DEMO_ZIP), "/root/demo.zip")
)
app = modal.App("faceunlock-gpu-benchmark")


@app.function(image=image, gpu="T4", cpu=2.0, memory=8192, timeout=1800)
def benchmark() -> dict:
    import sys
    import tempfile
    import time

    sys.path.insert(0, "/root")
    import tensorflow as tf

    from src.retrain import run_retrain, validate_upload

    out = {"gpus_visible_to_tensorflow": [gpu.name for gpu in tf.config.list_physical_devices("GPU")]}
    work = tempfile.mkdtemp()
    data = validate_upload("/root/demo.zip", f"{work}/data")
    for mode in ("light", "deep"):
        started = time.time()
        result = run_retrain(
            mode, data, "/root/models/faceunlock_v1.keras", f"{work}/{mode}.keras", lambda _: None,
            replay_zip="/root/models/replay_faces.zip", replay_dir=f"{work}/replay",
        )
        out[mode] = {
            "seconds": round(time.time() - started),
            "uploaded_people_auc": [round(result["old"]["auc"], 3), round(result["new"]["auc"], 3)],
            "original_people_auc": [round(result["replay_old"]["auc"], 3), round(result["replay_new"]["auc"], 3)],
        }
    return out


@app.local_entrypoint()
def main():
    print("RESULT", benchmark.remote())
