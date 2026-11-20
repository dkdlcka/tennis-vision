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

    assert report["media"] == {"condensed": True, "highlights": True}
    assert report["highlights"][0]["tags"][0] == "ace"
    for url in (f"/analyses/{job}/condensed.mp4", f"/analyses/{job}/highlights.mp4", f"/analyses/{job}/points/0.mp4"):
        r = client.get(url)
        assert r.status_code == 200 and r.headers["content-type"] == "video/mp4" and len(r.content) > 1000, url
    r = client.get(f"/analyses/{job}/points/0.mp4", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100
    assert client.get(f"/analyses/{job}/points/5.mp4").status_code == 404
