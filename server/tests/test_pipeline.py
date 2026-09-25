"""End to end: render a synthetic match to an .mp4 and analyze it like an upload."""

import pytest

from synth import Camera, simulate, standard_match, write_video
import cv2

from tennis_vision.pipeline import analyze_video
from tennis_vision.render import render_overlay


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("video") / "match.mp4")
    write_video(path, simulate(standard_match(), 30), Camera(), 30)
    return path


@pytest.fixture(scope="module")
def report(video):
    return analyze_video(video)


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


def test_shot_details(report):
    b = report["bounces"]
    assert b[0]["zone"] == "body"
    assert b[3]["miss"] == "long"
    assert b[4]["miss"] == "long"  # first serve of point 2 landed past the service line
    assert b[-1]["zone"] == "wide"
    assert report["stats"]["players"]["B"]["errors_long"] == 1
    assert report["stats"]["players"]["A"]["serve_zones"] == {"wide": 1, "body": 2, "T": 0}


def test_overlay_video(video, report, tmp_path):
    out = render_overlay(video, report, str(tmp_path / "overlay.mp4"))
    cap = cv2.VideoCapture(out)
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == report["video"]["frames"]
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == report["video"]["width"]
