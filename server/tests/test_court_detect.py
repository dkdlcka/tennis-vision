import numpy as np

from synth import Camera, render, simulate, standard_match
from tennis_vision.court import DOUBLES_CORNERS
from tennis_vision.court_detect import detect_court


def test_detects_synthetic_court():
    cam = Camera()
    frame = next(render(simulate(standard_match(), 30), cam))
    found = detect_court(frame)
    assert found is not None
    hom, score = found
    err = np.linalg.norm(hom.image_corners - cam.project(DOUBLES_CORNERS), axis=1)
    assert err.max() < 8, err
    assert score > 0.8


def test_blank_frame_has_no_court():
    assert detect_court(np.full((720, 1280, 3), (60, 120, 60), np.uint8)) is None
