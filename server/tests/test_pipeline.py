"""End to end: render a synthetic match to an .mp4 and analyze it like an upload."""

import pytest

from synth import Camera, simulate, standard_match, write_video
from tennis_vision.pipeline import analyze_video


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("video") / "match.mp4")
    write_video(path, simulate(standard_match(), 30), Camera(), 30)
    return analyze_video(path)


def test_court_found_automatically(report):
    assert report["court"]["source"] == "auto"


def test_points_and_score(report):
    outcomes = [(p["winner"], p["reason"]) for p in report["points"]]
    assert outcomes == [("A", "out"), (None, "fault"), ("B", "double_bounce"), ("A", "ace")]
    assert report["score"]["points"] == ["30", "15"]


def test_line_calls(report):
    calls = [(b["kind"], b["in"]) for b in report["bounces"]]
    assert calls == [
        ("serve", True),
        ("rally", True),
        ("rally", True),
        ("rally", False),
        ("serve", False),
        ("serve", True),
        ("rally", True),
        ("rally", False),
        ("serve", True),
    ]


def test_stats(report):
    a = report["stats"]["players"]["A"]
    assert a["aces"] == 1
    assert a["first_serves"] == 3 and a["first_serves_in"] == 2
    assert 80 < a["serve_speed_max_kmh"] < 140
