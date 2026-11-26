import random
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Callable

import numpy as np
from PIL import Image

from src.evaluation import embed_faces, evaluate_pairs, pair_scores
from src.preprocessing import build_pairs, load_training_face
from src.training import RunConfig, train

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MIN_PEOPLE = 10
MIN_PHOTOS = 2
MAX_FILES = 5000
MAX_BYTES = 500 * 1024 * 1024
EVAL_FRACTION = 0.3
EVAL_PAIRS = 400


def _ignored(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return (
        not parts
        or PurePosixPath(name).is_absolute()
        or any(part == ".." or part.startswith(".") or part == "__MACOSX" for part in parts)
    )


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def validate_upload(zip_path: str, dest_dir: str) -> dict[str, list[str]]:
    """Check a zip of `person/photo.jpg` files, extract it safely, return person -> photo paths."""
    try:
        archive = zipfile.ZipFile(zip_path)
    except (zipfile.BadZipFile, OSError):
        raise ValueError("The upload is not a valid zip file.") from None

    with archive:
        infos = [i for i in archive.infolist() if not i.is_dir() and not _ignored(i.filename)]
        if len(infos) > MAX_FILES:
            raise ValueError(f"The zip has more than {MAX_FILES} files.")
        if sum(i.file_size for i in infos) > MAX_BYTES:
            raise ValueError("The zip is larger than 500 MB when unpacked.")

        parts = [PurePosixPath(i.filename).parts for i in infos]
        if parts and all(len(p) >= 3 for p in parts) and len({p[0] for p in parts}) == 1:
            parts = [p[1:] for p in parts]  # drop one wrapper folder

        dest = Path(dest_dir)
        index: dict[str, list[str]] = {}
        for number, (info, p) in enumerate(zip(infos, parts)):
            if len(p) != 2 or Path(p[1]).suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            person_dir = dest / _safe(p[0])
            person_dir.mkdir(parents=True, exist_ok=True)
            target = person_dir / f"{number}_{_safe(p[1])}"
            try:
                target.write_bytes(archive.read(info))
                with Image.open(target) as image:
                    image.verify()
            except (zipfile.BadZipFile, OSError, SyntaxError):
                raise ValueError(f"{p[1]} is not a valid image.") from None
            index.setdefault(_safe(p[0]), []).append(str(target))

    singles = sum(1 for photos in index.values() if len(photos) < MIN_PHOTOS)
    index = {name: sorted(photos) for name, photos in index.items() if len(photos) >= MIN_PHOTOS}
    if len(index) < MIN_PEOPLE:
        raise ValueError(
            f"Need at least {MIN_PEOPLE} people with {MIN_PHOTOS}+ photos each; found {len(index)}"
            + (f" ({singles} folders had only one photo)." if singles else ".")
        )
    return index


def _metrics(model, faces: np.ndarray, rows_a, rows_b, labels) -> dict:
    scores = pair_scores(embed_faces(model, faces), rows_a, rows_b)
    result = evaluate_pairs(scores, labels)
    result.pop("fpr")
    result.pop("tpr")
    return result


def run_retrain(
    mode: str,
    data: dict[str, list[str]],
    base_model_path: str,
    out_path: str,
    progress: Callable[[float], None],
    epochs: int = 10,
    seed: int = 0,
) -> dict:
    """Fine-tune the current model on uploaded people and compare old vs new on held-out people.

    light: backbone frozen, only the embedding layer learns.
    deep:  backbone unfrozen (BatchNorm stays frozen) and fine-tuned as well.
    """
    if mode not in ("light", "deep"):
        raise ValueError("mode must be 'light' or 'deep'")
    from tensorflow import keras

    rng = random.Random(seed)
    people = sorted(data)
    rng.shuffle(people)
    n_eval = max(3, int(len(people) * EVAL_FRACTION))
    eval_people, train_people = people[:n_eval], people[n_eval:]
    progress(0.02)

    train_paths = [p for name in train_people for p in data[name]]
    x_train = np.stack([load_training_face(p) for p in train_paths])
    y_train = np.array([i for i, name in enumerate(train_people) for _ in data[name]])
    eval_paths = sorted({p for name in eval_people for p in data[name]})
    x_eval = np.stack([load_training_face(p) for p in eval_paths])
    row = {p: i for i, p in enumerate(eval_paths)}
    pairs = build_pairs({n: data[n] for n in eval_people}, EVAL_PAIRS, seed)
    rows_a = np.array([row[a] for a, _, _ in pairs])
    rows_b = np.array([row[b] for _, b, _ in pairs])
    labels = np.array([label for _, _, label in pairs])
    progress(0.08)

    old = _metrics(keras.models.load_model(base_model_path), x_eval, rows_a, rows_b, labels)

    deep = mode == "deep"
    config = RunConfig(
        name=mode,
        augment=True,
        early_stopping=True,
        fine_tune=deep,
        unfreeze_layers=10_000,
        learning_rates=(1e-4, 1e-5),
    )
    total_epochs = epochs * (2 if deep else 1)
    finished = [0]

    def on_epoch_end(_epoch, _logs=None):
        finished[0] += 1
        progress(0.1 + 0.8 * min(finished[0] / total_epochs, 1.0))

    embedding = keras.models.load_model(base_model_path)
    train(
        config,
        (x_train, y_train),
        (x_eval, rows_a, rows_b, labels),
        n_classes=len(train_people),
        epochs=epochs,
        embedding=embedding,
        extra_callbacks=[keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)],
    )
    progress(0.95)
    new = _metrics(embedding, x_eval, rows_a, rows_b, labels)
    embedding.save(out_path)
    progress(1.0)
    return {
        "mode": mode,
        "people_train": len(train_people),
        "people_eval": len(eval_people),
        "eval_pairs": len(pairs),
        "old": old,
        "new": new,
    }
