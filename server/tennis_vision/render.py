"""Draw an analysis report onto its video: court lines, ball trail, line calls, score.

Writes H.264 (playable on phones) when an ffmpeg binary is available, either on
PATH or from the optional imageio-ffmpeg package, and falls back to OpenCV's
mp4v otherwise.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .court import COURT_LINES, CourtHomography
from .stabilize import apply

TRAIL_FRAMES = 12
CALL_SECONDS = 1.2
RESULT_SECONDS = 2.5
REASON_TEXT = {
    "ace": "ACE",
    "fault": "FAULT",
    "double_fault": "DOUBLE FAULT",
    "out": "OUT",
    "double_bounce": "DOUBLE BOUNCE",
    "winner": "WINNER",
}
GREEN, RED, YELLOW, WHITE, DARK = (80, 175, 76), (57, 57, 229), (0, 230, 255), (255, 255, 255), (31, 48, 18)


def _ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def _label(name: str, fallback: str) -> str:
    # OpenCV's built-in fonts only draw ASCII.
    return name if name.isascii() and name.strip() else fallback


class _Writer:
    def __init__(self, path: str, fps: float, size: tuple[int, int], audio_from: str):
        self.proc = None
        ffmpeg = _ffmpeg()
        if ffmpeg:
            w, h = size
            cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24"]
            cmd += ["-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-", "-i", audio_from]
            cmd += ["-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23"]
            cmd += ["-c:a", "aac", "-shortest", "-movflags", "+faststart", path]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            self.cv = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, frame: np.ndarray) -> None:
        if self.proc:
            self.proc.stdin.write(frame.tobytes())
        else:
            self.cv.write(frame)

    def close(self) -> None:
        if self.proc:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise RuntimeError("ffmpeg failed to write the overlay video")
        else:
            self.cv.release()


def render_overlay(video_path: str, report: dict, out_path: str) -> str:
    cap = cv2.VideoCapture(video_path)
    fps = report["video"]["fps"]
    ok, frame = cap.read()
    if not ok:
        raise ValueError("video has no frames")
    h, w = frame.shape[:2]
    s = max(w, h) / 1280  # scale drawing to the video size

    court = CourtHomography(np.array(report["court"]["corners_px"]))
    court_pts = [court.to_image(np.array([a, b])) for a, b in COURT_LINES]
    transforms = report.get("frame_transforms")
    track = {int(f): (x, y) for f, x, y in report["ball_track"]}
    names = {p: _label(n, p) for p, n in report["players"].items()}
    points = report["points"]
    bounces = report["bounces"]

    writer = _Writer(out_path, fps, (w, h), video_path)
    i = 0
    while ok:
        t = i / fps
        overlay = frame.copy()
        if transforms and i < len(transforms):
            m = np.array(transforms[i]).reshape(3, 3)
            court_lines = [apply(m, p).round().astype(int) for p in court_pts]
        else:
            court_lines = [p.round().astype(int) for p in court_pts]
        for p in court_lines:
            cv2.line(overlay, tuple(p[0]), tuple(p[1]), YELLOW, max(1, int(s)), cv2.LINE_AA)
        frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

        trail = [track[f] for f in range(i - TRAIL_FRAMES, i + 1) if f in track]
        for a, b in zip(trail, trail[1:]):
            cv2.line(frame, _pt(a), _pt(b), YELLOW, max(2, int(2 * s)), cv2.LINE_AA)
        if i in track:
            cv2.circle(frame, _pt(track[i]), int(8 * s), YELLOW, max(2, int(2 * s)), cv2.LINE_AA)

        for b in bounces:
            if 0 <= t - b["t"] < CALL_SECONDS:
                color = GREEN if b["in"] else RED
                cm = abs(b["margin_cm"])
                text = ("IN" if b["in"] else "OUT") + (f" {cm:.0f}cm" if cm < 100 else "")
                c = _pt(b["image_xy"])
                cv2.circle(frame, c, int(14 * s), color, max(3, int(3 * s)), cv2.LINE_AA)
                _tag(frame, text, (c[0] + int(18 * s), c[1] - int(18 * s)), color, s)

        done = [p for p in points if p["end_t"] <= t]
        score = done[-1]["score_after"] if done else None
        _scoreboard(frame, score, names, s)
        if done and t - done[-1]["end_t"] < RESULT_SECONDS:
            p = done[-1]
            who = f"{names[p['winner']]} WINS POINT - " if p["winner"] else ""
            speed = f"  serve {p['serve_speed_kmh']:.0f} km/h" if p.get("serve_speed_kmh") else ""
            _tag(
                frame,
                who + REASON_TEXT.get(p["reason"], p["reason"]) + speed,
                (int(20 * s), h - int(30 * s)),
                DARK,
                s * 1.2,
            )

        writer.write(frame)
        ok, frame = cap.read()
        i += 1
    cap.release()
    writer.close()
    return out_path


def _pt(p) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def _tag(frame: np.ndarray, text: str, origin: tuple[int, int], color, s: float) -> None:
    scale = 0.7 * s
    thick = max(1, int(2 * s))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = origin
    pad = int(6 * s)
    cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + base + pad // 2), color, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, thick, cv2.LINE_AA)


def _scoreboard(frame: np.ndarray, score: dict | None, names: dict, s: float) -> None:
    rows = []
    for k, p in enumerate(("A", "B")):
        sets = " ".join(str(st[k]) for st in score["sets"]) if score else "0"
        pts = score["points"][k] if score else "0"
        serving = score and score["server"] == p
        rows.append((("* " if serving else "  ") + names[p], sets, pts))
    scale, thick = 0.7 * s, max(1, int(2 * s))
    line_h = int(30 * s)
    x0, y0 = int(16 * s), int(16 * s)
    width = int(300 * s)
    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + line_h * 2 + int(12 * s)), DARK, -1)
    for r, (name, sets, pts) in enumerate(rows):
        y = y0 + int(8 * s) + line_h * (r + 1) - int(8 * s)
        cv2.putText(frame, name[:14], (x0 + int(8 * s), y), cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, thick, cv2.LINE_AA)
        cv2.putText(frame, sets, (x0 + int(180 * s), y), cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, thick, cv2.LINE_AA)
        cv2.putText(frame, pts, (x0 + int(240 * s), y), cv2.FONT_HERSHEY_SIMPLEX, scale, YELLOW, thick, cv2.LINE_AA)


def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Draw an analysis report onto its video")
    ap.add_argument("video")
    ap.add_argument("report")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or str(Path(args.video).with_name(Path(args.video).stem + "_analyzed.mp4"))
    render_overlay(args.video, json.loads(Path(args.report).read_text()), out)
    print(out)


if __name__ == "__main__":
    main()
