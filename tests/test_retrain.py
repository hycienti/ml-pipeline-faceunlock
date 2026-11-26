import io
import zipfile

import numpy as np
import pytest
from PIL import Image

from src.model import build_embedding_model
from src.retrain import run_retrain, validate_upload
from tests.test_replay import make_replay_zip


def png_bytes(seed=0, size=64):
    rng = np.random.default_rng(seed)
    buffer = io.BytesIO()
    Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8)).save(buffer, "PNG")
    return buffer.getvalue()


def make_zip(path, layout, extra=None):
    """layout: {person: n_photos}. Files are written as person/photo_i.png."""
    with zipfile.ZipFile(path, "w") as zf:
        for person, n in layout.items():
            for i in range(n):
                zf.writestr(f"{person}/photo_{i}.png", png_bytes(hash((person, i)) % 1000))
        for name, data in (extra or {}).items():
            zf.writestr(name, data)
    return path


def ten_people(n=2):
    return {f"person{i}": n for i in range(10)}


def test_valid_upload_returns_identity_to_photos(tmp_path):
    zip_path = make_zip(tmp_path / "ok.zip", ten_people())
    index = validate_upload(str(zip_path), str(tmp_path / "out"))
    assert len(index) == 10
    assert all(len(photos) == 2 for photos in index.values())
    assert all(p.endswith(".png") for photos in index.values() for p in photos)


def test_single_wrapper_folder_is_stripped(tmp_path):
    with zipfile.ZipFile(tmp_path / "w.zip", "w") as zf:
        for person, n in ten_people().items():
            for i in range(n):
                zf.writestr(f"faces/{person}/{i}.png", png_bytes(i))
    assert len(validate_upload(str(tmp_path / "w.zip"), str(tmp_path / "out"))) == 10


def test_not_a_zip_is_rejected(tmp_path):
    bad = tmp_path / "notzip.zip"
    bad.write_bytes(b"this is not a zip")
    with pytest.raises(ValueError, match="valid zip"):
        validate_upload(str(bad), str(tmp_path / "out"))


def test_truncated_zip_is_rejected(tmp_path):
    good = make_zip(tmp_path / "ok.zip", ten_people())
    cut = tmp_path / "cut.zip"
    cut.write_bytes(good.read_bytes()[:200])
    with pytest.raises(ValueError, match="valid zip"):
        validate_upload(str(cut), str(tmp_path / "out"))


def test_single_photo_folders_do_not_count(tmp_path):
    layout = {**{f"p{i}": 2 for i in range(5)}, **{f"solo{i}": 1 for i in range(8)}}
    zip_path = make_zip(tmp_path / "few.zip", layout)
    with pytest.raises(ValueError, match="at least 10 people") as error:
        validate_upload(str(zip_path), str(tmp_path / "out"))
    assert "found 5" in str(error.value)


def test_corrupt_image_is_rejected_with_its_name(tmp_path):
    zip_path = make_zip(tmp_path / "c.zip", ten_people(), extra={"person0/broken.png": b"junk"})
    with pytest.raises(ValueError, match="broken.png"):
        validate_upload(str(zip_path), str(tmp_path / "out"))


def test_macos_junk_and_path_traversal_are_ignored(tmp_path):
    extra = {
        "__MACOSX/person0/._photo_0.png": b"junk",
        "../../evil/x.png": png_bytes(),
        "person0/.DS_Store": b"junk",
    }
    zip_path = make_zip(tmp_path / "j.zip", ten_people(), extra=extra)
    out = tmp_path / "sandbox" / "out"
    index = validate_upload(str(zip_path), str(out))
    assert len(index) == 10
    assert not (tmp_path / "evil").exists() and not (tmp_path / "sandbox" / "evil").exists()


@pytest.mark.parametrize("mode", ["light", "deep"])
def test_run_retrain_returns_old_and_new_metrics_and_saves_model(tmp_path, mode):
    base = tmp_path / "base.keras"
    build_embedding_model().save(base)
    zip_path = make_zip(tmp_path / "up.zip", ten_people(3))
    data = validate_upload(str(zip_path), str(tmp_path / "data"))

    progress = []
    result = run_retrain(
        mode, data, str(base), str(tmp_path / "new.keras"), progress.append, epochs=1
    )

    assert (tmp_path / "new.keras").exists()
    assert result["mode"] == mode
    assert 0.0 <= result["old"]["auc"] <= 1.0 and 0.0 <= result["new"]["auc"] <= 1.0
    assert "threshold" in result["new"]
    assert progress[-1] == 1.0 and progress == sorted(progress)


def test_run_retrain_rejects_unknown_mode(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        run_retrain("medium", {}, "a", "b", lambda _: None)


def test_run_retrain_with_replay_trains_on_old_and_new_people_and_checks_forgetting(tmp_path):
    base = tmp_path / "base.keras"
    build_embedding_model().save(base)
    data = validate_upload(str(make_zip(tmp_path / "up.zip", ten_people(3))), str(tmp_path / "data"))
    replay_zip = make_replay_zip(tmp_path / "replay.zip", train_people=12, val_people=4)

    result = run_retrain(
        "light", data, str(base), str(tmp_path / "new.keras"), lambda _: None, epochs=1,
        replay_zip=str(replay_zip), replay_dir=str(tmp_path / "replay"),
    )

    assert result["replay_people"] == 12 and result["people_train"] == 7
    assert result["classes"] == 19  # 12 original + 7 newly uploaded people
    for key in ("replay_old", "replay_new"):
        assert 0.0 <= result[key]["auc"] <= 1.0
    assert result["login_threshold"]["far_target"] == 0.02  # chosen on the original validation pairs


def test_run_retrain_without_replay_has_no_forgetting_check(tmp_path):
    base = tmp_path / "base.keras"
    build_embedding_model().save(base)
    data = validate_upload(str(make_zip(tmp_path / "up.zip", ten_people(3))), str(tmp_path / "data"))
    result = run_retrain("light", data, str(base), str(tmp_path / "new.keras"), lambda _: None, epochs=1)
    assert "replay_old" not in result and "login_threshold" not in result
