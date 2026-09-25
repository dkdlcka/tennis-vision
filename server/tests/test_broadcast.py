"""Broadcast-style footage: court shots cut with close-ups, and a camera that pans and zooms."""

import cv2
import numpy as np
import pytest

from synth import Camera, render_moving, simulate, standard_match
from tennis_vision.broadcast import analyze_broadcast, find_court_shots

FPS = 30
CUT = 45  # frames of close-up between the court shots


def _close_up(n: int, seed: int):
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.integers(0, 255, (720, 1280, 3)).astype(np.uint8), (0, 0), 25)
    base = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX)
    for i in range(n):
        yield np.roll(base, 3 * i, axis=1)


def _panning(i: int) -> Camera:
    return Camera(target=(1.5 * np.sin(i / 40), -3.0, 0.0), hfov_deg=60 + 3 * np.sin(i / 55))


@pytest.fixture(scope="module")
def broadcast(tmp_path_factory):
    # A rally won by the server, then the next point's first serve, a fault.
    points = standard_match()
    rally, fault = simulate(points[:1], FPS), simulate(points[1:2], FPS)
    path = str(tmp_path_factory.mktemp("video") / "broadcast.mp4")
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1280, 720))
    shots, bounces, t = [], [], 0
    for k, truth in enumerate((rally, fault)):
        for frame in _close_up(CUT, k):
            out.write(frame)
        t += CUT
        for frame in render_moving(truth, lambda i, t=t: _panning(t + i)):
            out.write(frame)
        shots.append((t, t + len(truth.positions)))
        bounces += [(f + t, x, y) for f, x, y in truth.bounces]
        t += len(truth.positions)
    out.release()
    return path, shots, bounces


def test_finds_court_shots(broadcast):
    path, truth_shots, _ = broadcast
    shots, fps, _ = find_court_shots(path)
    assert len(shots) == len(truth_shots)
    for shot, (a, b) in zip(shots, truth_shots):
        # Sampled twice a second, so the edges land within half a second.
        assert abs(shot.start - a) <= fps / 2 and abs(shot.end - b) <= fps / 2, (shot, a, b)


def test_calls_through_a_moving_camera(broadcast):
    path, _, truth_bounces = broadcast
    report = analyze_broadcast(path)
    found = [(b["t"] * FPS, *b["court_xy"]) for b in report["bounces"]]
    errors = []
    for f, x, y in truth_bounces:
        near = [b for b in found if abs(b[0] - f) <= 2]
        assert near, f"bounce at frame {f} missed"
        errors.append(np.hypot(near[0][1] - x, near[0][2] - y))
    # A landing as the ball leaves the bottom of the picture is extrapolated, so less exact.
    assert np.median(errors) < 0.15 and max(errors) < 1.0, errors
    assert [p["reason"] for p in report["points"]] == ["out", "fault"]
