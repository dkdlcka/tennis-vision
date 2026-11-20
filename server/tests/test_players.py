import numpy as np
import pytest

from synth import Camera, SynthPlayers, player_x, simulate, standard_match, write_video
from tennis_vision.court import DOUBLES_CORNERS, CourtHomography, Side
from tennis_vision.pipeline import analyze_video
from tennis_vision.players import PlayerTracker


def test_tracker_keeps_one_player_per_half():
    cam = Camera()
    hom = CourtHomography(cam.project(DOUBLES_CORNERS))
    det = SynthPlayers(cam)
    tracker = PlayerTracker(hom)
    for f in range(0, 90, 3):
        boxes = det.detect(None)
        # A ball kid standing far outside the court should be ignored.
        kid = cam.project([(8.0, -3.0, 0)])[0]
        boxes.append((kid[0] - 5, kid[1] - 40, kid[0] + 5, kid[1], 0.9))
        tracker.update(f, boxes)
    near = tracker.tracks[Side.NEAR]
    assert len(near.frames) == 30
    for f, (x, _) in zip(near.frames, near.xy, strict=True):
        assert x == pytest.approx(player_x(f, True), abs=0.15)
    assert near.at(4.5)[0] == pytest.approx(np.interp(4.5, [3, 6], [player_x(3, True), player_x(6, True)]), abs=0.15)


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("video") / "match.mp4")
    cam = Camera()
    write_video(path, simulate(standard_match(), 30), cam, 30)
    return analyze_video(path, player_detector=SynthPlayers(cam))


def test_shots(report):
    shots = report["shots"]
    assert [(s["point"], s["player"], s["type"]) for s in shots] == [
        (0, "A", "serve"),
        (0, "B", "forehand"),
        (0, "A", "forehand"),
        (0, "B", "forehand"),
        (1, "A", "serve"),
        (2, "A", "serve"),
        (2, "B", "backhand"),
        (3, "A", "serve"),
    ]
    assert [s["in"] for s in shots] == [True, True, True, False, False, True, True, True]
    assert all(s["speed_kmh"] for s in shots)
    assert shots[2]["depth"] == "deep" and shots[6]["depth"] == "short"
    assert report["points"][0]["shot_ids"] == [0, 1, 2, 3]


def test_stats_by_type_and_movement(report):
    a = report["stats"]["players"]["A"]
    assert a["by_type"]["serve"]["count"] == 4 and a["by_type"]["serve"]["in_pct"] == 75.0
    # The synthetic players sway ±2.5 m; over the points that adds up to some meters.
    assert 5 < a["movement"]["distance_m"] < 60
    assert a["movement"]["positions"]
