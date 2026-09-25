import cv2
import numpy as np

from synth import Camera, render, simulate, standard_match
from tennis_vision.court import DOUBLES_CORNERS
from tennis_vision.segments import cut, find_court_segments

FPS = 30


def test_main_camera_runs_are_found(tmp_path):
    main = Camera()
    replay = Camera(position=(12.0, -8.0, 4.0), target=(0.0, 2.0, 0.0))  # another angle
    truth = simulate(standard_match()[:1], FPS)
    frames = list(render(truth, main))[: 6 * FPS]
    path = str(tmp_path / "broadcast.mp4")
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (main.width, main.height))
    for f in frames:  # 0-6 s main camera
        out.write(f)
    for f in list(render(truth, replay))[: 3 * FPS]:  # 6-9 s replay angle
        out.write(f)
    for _ in range(2 * FPS):  # 9-11 s crowd shot stand-in
        out.write(np.full_like(frames[0], 90))
    for f in frames[: 6 * FPS]:  # 11-17 s main camera again
        out.write(f)
    out.release()

    segs = find_court_segments(path, min_len_s=3)
    assert [(round(s.start), round(s.end)) for s in segs] == [(0, 6), (11, 17)]
    err = np.abs(np.array(segs[0].corners) - main.project(DOUBLES_CORNERS)).max()
    assert err < 8

    clip = cut(path, segs[1], str(tmp_path / "clip.mp4"))
    cap = cv2.VideoCapture(clip)
    assert abs(cap.get(cv2.CAP_PROP_FRAME_COUNT) / FPS - 6) < 0.6
