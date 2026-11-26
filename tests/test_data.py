from pathlib import Path

import pytest

from src.data import build_lfw_splits, export_splits


def make_fake_lfw(root: Path, people: dict[str, int], official_test: list[str]):
    faces = root / "lfw-deepfunneled" / "lfw-deepfunneled"
    for name, n in people.items():
        (faces / name).mkdir(parents=True)
        for i in range(1, n + 1):
            (faces / name / f"{name}_{i:04d}.jpg").touch()
    lines = ["name,images"] + [f"{name},1" for name in official_test]
    (root / "peopleDevTest.csv").write_text("\n".join(lines))


@pytest.fixture
def fake_lfw(tmp_path):
    people = {f"p{i:02d}": 2 + (i % 4) for i in range(40)}
    people.update({"solo_a": 1, "solo_b": 1, "bigstar": 60, "official_x": 5, "official_y": 3})
    make_fake_lfw(tmp_path, people, official_test=["official_x", "official_y"])
    return tmp_path


def test_splits_are_disjoint_and_exclude_singles_and_official_test_people(fake_lfw):
    splits = build_lfw_splits(fake_lfw)
    everyone = [set(splits[k]) for k in ("train", "val", "test")]
    assert not (everyone[0] & everyone[1] or everyone[0] & everyone[2] or everyone[1] & everyone[2])
    used = set().union(*everyone)
    assert "solo_a" not in used and "solo_b" not in used
    assert "official_x" not in used and "official_y" not in used
    assert len(used) == 41  # 40 regular people + bigstar


def test_split_proportions_and_photo_cap(fake_lfw):
    splits = build_lfw_splits(fake_lfw, max_photos=20)
    assert (len(splits["test"]), len(splits["val"]), len(splits["train"])) == (8, 4, 29)
    assert all(len(photos) <= 20 for k in splits for photos in splits[k].values())
    all_photos = [p for k in splits for photos in splits[k].values() for p in photos]
    assert len(all_photos) == len(set(all_photos))


def test_splits_are_repeatable_and_seed_dependent(fake_lfw):
    assert build_lfw_splits(fake_lfw, seed=1) == build_lfw_splits(fake_lfw, seed=1)
    assert build_lfw_splits(fake_lfw, seed=1) != build_lfw_splits(fake_lfw, seed=2)


REAL_LFW = Path(__file__).parent.parent / "data/lfw"


def test_real_lfw_matches_the_numbers_reported_in_the_notebook():
    if not REAL_LFW.exists():
        pytest.skip("LFW not downloaded")
    splits = build_lfw_splits(REAL_LFW)
    counts = {k: (len(v), sum(map(len, v.values()))) for k, v in splits.items()}
    assert counts == {"train": (830, 3505), "val": (118, 530), "test": (236, 1186)}


def test_export_splits_copies_each_photo_into_split_and_person_folders(fake_lfw, tmp_path):
    splits = build_lfw_splits(fake_lfw)
    counts = export_splits(splits, tmp_path / "data")
    for split, index in splits.items():
        assert counts[split] == sum(len(photos) for photos in index.values())
        for person, photos in index.items():
            copied = sorted(p.name for p in (tmp_path / "data" / split / person).glob("*.jpg"))
            assert copied == sorted(Path(p).name for p in photos)


def test_export_splits_can_be_rerun_without_duplicating(fake_lfw, tmp_path):
    splits = build_lfw_splits(fake_lfw)
    first = export_splits(splits, tmp_path / "data")
    again = export_splits(splits, tmp_path / "data")
    assert first == again
    assert len(list((tmp_path / "data").rglob("*.jpg"))) == sum(first.values())
