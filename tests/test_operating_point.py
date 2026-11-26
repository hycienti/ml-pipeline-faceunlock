from src.model import build_embedding_model
from src.operating_point import login_threshold
from src.replay import load_replay
from tests.test_replay import make_replay_zip


def test_login_threshold_reports_the_operating_point(tmp_path):
    replay = load_replay(str(make_replay_zip(tmp_path / "r.zip", 6, 6)), str(tmp_path / "out"))
    result = login_threshold(build_embedding_model(), replay, far=0.1)
    assert set(result) == {"threshold", "far_target", "far", "frr"}
    assert result["far_target"] == 0.1 and result["far"] <= 0.1
    assert -1.0 <= result["threshold"] <= 1.5 and 0.0 <= result["frr"] <= 1.0
