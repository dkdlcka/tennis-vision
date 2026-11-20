"""HTTP API the mobile app talks to.

POST /court/detect   a still frame -> detected court corners, for the app to confirm
POST /analyses       upload a video (+ options) -> analysis id
GET  /analyses/{id}  status, progress and, when done, the full report
GET  /analyses/{id}/condensed.mp4      only the points, dead time removed
GET  /analyses/{id}/highlights.mp4     the best points joined into one reel
GET  /analyses/{id}/points/{n}.mp4     one point's clip (cut on first request)
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .court_detect import detect_court
from .highlights import cut_video, ffmpeg_available, point_segment, reel_segments
from .pipeline import CourtNotFound, Options, analyze_video

DATA_DIR = Path(os.environ.get("TENNIS_DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Tennis Vision")
_executor = ThreadPoolExecutor(max_workers=int(os.environ.get("TENNIS_WORKERS", "1")))
_jobs: dict[str, dict] = {}
_lock = threading.Lock()


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/court/detect")
async def court_detect(image: UploadFile = File(...)) -> dict:
    data = np.frombuffer(await image.read(), np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "not an image")
    found = detect_court(frame)
    if found is None:
        return {"found": False}
    hom, score = found
    return {"found": True, "score": round(score, 3), "corners": hom.image_corners.round(1).tolist()}


@app.post("/analyses")
async def create_analysis(video: UploadFile = File(...), options: str = Form("{}")) -> dict:
    try:
        opts = Options(**json.loads(options))
    except (TypeError, ValueError) as e:
        raise HTTPException(400, f"bad options: {e}")
    job_id = uuid.uuid4().hex[:12]
    job_dir = DATA_DIR / job_id
    job_dir.mkdir()
    suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
    path = job_dir / f"video{suffix}"
    with path.open("wb") as f:
        while chunk := await video.read(1 << 20):
            f.write(chunk)
    with _lock:
        _jobs[job_id] = {"id": job_id, "status": "queued", "progress": 0.0}
    _executor.submit(_run, job_id, str(path), opts)
    return {"id": job_id}


@app.get("/analyses/{job_id}")
def get_analysis(job_id: str) -> dict:
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        result = DATA_DIR / job_id / "report.json"
        if result.exists():
            return {"id": job_id, "status": "done", "progress": 1.0, "report": json.loads(result.read_text())}
        raise HTTPException(404, "unknown analysis")
    if job["status"] == "done":
        return {**job, "report": json.loads((DATA_DIR / job_id / "report.json").read_text())}
    return job


def _run(job_id: str, path: str, opts: Options) -> None:
    def progress(p: float) -> None:
        with _lock:
            _jobs[job_id]["progress"] = round(p, 3)

    with _lock:
        _jobs[job_id]["status"] = "running"
    try:
        report = analyze_video(path, opts, progress)
        report["media"] = _cut_media(Path(path), report)
        (Path(path).parent / "report.json").write_text(json.dumps(report, ensure_ascii=False))
        with _lock:
            _jobs[job_id].update(status="done", progress=1.0)
    except CourtNotFound as e:
        with _lock:
            _jobs[job_id].update(status="error", error="court_not_found", message=str(e))
    except Exception as e:  # report every failure to the app instead of hanging
        with _lock:
            _jobs[job_id].update(status="error", error="analysis_failed", message=str(e))


def _cut_media(src: Path, report: dict) -> dict:
    """Cuts the condensed video and the highlight reel next to the upload. Never fails the analysis."""
    media = {"condensed": False, "highlights": False}
    if not ffmpeg_available():
        return media
    jobs = {"condensed": report["condensed"]["segments"], "highlights": reel_segments(report)}
    for name, segments in jobs.items():
        if not segments:
            continue
        try:
            cut_video(src, segments, src.parent / f"{name}.mp4")
            media[name] = True
        except (RuntimeError, ValueError):
            pass
    return media


def _job_dir(job_id: str) -> Path:
    if not job_id.isalnum():
        raise HTTPException(404, "unknown analysis")
    d = DATA_DIR / job_id
    if not (d / "report.json").exists():
        raise HTTPException(404, "analysis not finished")
    return d


@app.get("/analyses/{job_id}/condensed.mp4")
def condensed_video(job_id: str) -> FileResponse:
    path = _job_dir(job_id) / "condensed.mp4"
    if not path.exists():
        raise HTTPException(404, "no condensed video")
    return FileResponse(path, media_type="video/mp4")


@app.get("/analyses/{job_id}/highlights.mp4")
def highlights_video(job_id: str) -> FileResponse:
    path = _job_dir(job_id) / "highlights.mp4"
    if not path.exists():
        raise HTTPException(404, "no highlights")
    return FileResponse(path, media_type="video/mp4")


@app.get("/analyses/{job_id}/points/{index}.mp4")
def point_clip(job_id: str, index: int) -> FileResponse:
    d = _job_dir(job_id)
    report = json.loads((d / "report.json").read_text())
    if not 0 <= index < len(report["points"]):
        raise HTTPException(404, "no such point")
    path = d / "clips" / f"point{index:03d}.mp4"
    if not path.exists():
        src = next((p for p in d.glob("video.*")), None)
        if src is None or not ffmpeg_available():
            raise HTTPException(404, "clip not available")
        with _clip_lock:
            if not path.exists():
                try:
                    cut_video(src, [point_segment(report, index)], path)
                except RuntimeError as e:
                    raise HTTPException(500, str(e))
    return FileResponse(path, media_type="video/mp4")


_clip_lock = threading.Lock()
