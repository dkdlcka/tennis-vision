import time

import cv2
from fastapi.testclient import TestClient

from synth import Camera, render, simulate, standard_match, write_video


def test_detect_and_analyze(tmp_path, monkeypatch):
    monkeypatch.setenv("TENNIS_DATA_DIR", str(tmp_path / "data"))
    import importlib

    import tennis_vision.api as api

    importlib.reload(api)
    client = TestClient(api.app)

    cam = Camera()
    truth = simulate(standard_match()[-1:], 30)
    frame = next(render(truth, cam))
    ok, jpg = cv2.imencode(".jpg", frame)
    r = client.post("/court/detect", files={"image": ("f.jpg", jpg.tobytes(), "image/jpeg")})
    assert r.status_code == 200 and r.json()["found"]

    video = tmp_path / "clip.mp4"
    write_video(str(video), truth, cam, 30)
    corners = r.json()["corners"]
    with video.open("rb") as f:
        r = client.post(
            "/analyses",
            files={"video": ("clip.mp4", f, "video/mp4")},
            data={"options": '{"corners": %s, "player_names": ["준호", "상대"]}' % corners},
        )
    job = r.json()["id"]
    for _ in range(120):
        status = client.get(f"/analyses/{job}").json()
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    assert status["status"] == "done", status
    report = status["report"]
    assert report["court"]["source"] == "manual"
    assert report["players"]["A"] == "준호"
    assert [p["reason"] for p in report["points"]] == ["ace"]


def test_rally_highlights(tmp_path, monkeypatch):

    from synth import render_moving
    from test_broadcast import _close_up, _panning

    monkeypatch.setenv("TENNIS_DATA_DIR", str(tmp_path / "data"))
    import importlib

    import tennis_vision.api as api

    importlib.reload(api)
    client = TestClient(api.app)

    # Close-up, then a served rally from a panning camera.
    video = tmp_path / "match.mp4"
    out = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    for frame in _close_up(45, 0):
        out.write(frame)
    for frame in render_moving(simulate(standard_match()[:1], 30), lambda i: _panning(45 + i)):
        out.write(frame)
    out.release()

    with video.open("rb") as f:
        job = client.post("/rallies", files={"video": ("match.mp4", f, "video/mp4")}).json()["id"]
    for _ in range(240):
        status = client.get(f"/rallies/{job}").json()
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    assert status["status"] == "done", status
    report = status["report"]
    assert report["stats"]["points"] == 1
    (point,) = report["points"]
    assert [b["in"] for b in point["bounces"]] == [True, True, True]
    assert abs(point["bounces"][0]["court_xy"][1] - 5.0) < 0.5

    r = client.get(f"/rallies/{job}/video")
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(r.content)
    cap = cv2.VideoCapture(str(clip))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert abs(n / 30 - point["clip_end_t"]) < 0.2
