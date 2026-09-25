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
