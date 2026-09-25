"""Find the stretches of a long video that show the main court camera.

Broadcasts and phone recordings mix rallies with close-ups, replays, crowd shots
and breaks between points. Sampling a few frames per second and keeping the
runs where the court is found in the same place isolates the playable footage,
which can then be cut into clips and analyzed with fixed court corners.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .court_detect import detect_court
from .render import _ffmpeg

SAMPLE_WIDTH = 960


@dataclass
class Segment:
    start: float  # seconds
    end: float
    corners: list[list[float]]  # doubles corners in source pixels

    @property
    def duration(self) -> float:
        return self.end - self.start


def find_court_segments(
    path: str,
    step_s: float = 0.5,
    min_len_s: float = 4.0,
    max_shift_frac: float = 0.05,
    min_score: float = 0.9,
    max_miss: int = 3,
) -> list[Segment]:
    """Runs of sampled frames showing the court from behind a baseline, holding still.

    A sample counts when the court is found with a wide, level near baseline (the
    usual main camera), and a run continues while the court stays within
    `max_shift_frac` of the frame width of where the run has had it. Up to
    `max_miss` samples in a row may fail (a player blocking a line) without
    ending the run. The analysis follows small pans and zooms inside a clip.
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(step_s * fps)))
    samples: list[tuple[float, np.ndarray | None]] = []
    frame_w = frame_h = 0
    scale = 1.0
    f = -1
    while True:
        # Grabbing without decoding is much faster than seeking to each sample.
        if not cap.grab():
            break
        f += 1
        if f % step:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        frame_h, frame_w = frame.shape[:2]
        scale = min(1.0, SAMPLE_WIDTH / frame_w)
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
        found = detect_court(small)
        corners = found[0].image_corners / scale if found and found[1] >= min_score else None
        samples.append((f / fps, corners))
    cap.release()

    # Keep only the behind-the-baseline view: a wide, level near baseline. Side
    # angles and replays from elsewhere in the stadium fail this even with a court found.
    def main_view(c: np.ndarray | None) -> bool:
        if c is None:
            return False
        near_w = c[1, 0] - c[0, 0]
        level = abs(c[1, 1] - c[0, 1]) < 0.04 * frame_h and abs(c[2, 1] - c[3, 1]) < 0.04 * frame_h
        return near_w > 0.4 * frame_w and level

    # A run lasts while the court stays put: the camera may zoom between points,
    # so each run is compared with itself rather than one global position.
    tol = max_shift_frac * frame_w
    segments: list[Segment] = []
    run: list[tuple[float, np.ndarray]] = []

    def close_run() -> None:
        if run and run[-1][0] - run[0][0] + step_s >= min_len_s:
            corners = np.median(np.array([c for _, c in run]), axis=0)
            segments.append(Segment(run[0][0], run[-1][0] + step_s, corners.round(1).tolist()))

    misses = 0
    for t, c in samples:
        if main_view(c) and (not run or np.abs(c - np.median([r for _, r in run], axis=0)).max() < tol):
            run.append((t, c))
            misses = 0
            continue
        if run and misses < max_miss:
            misses += 1
            continue
        close_run()
        run = [(t, c)] if main_view(c) else []
        misses = 0
    close_run()
    return segments


def cut(path: str, segment: Segment, out_path: str) -> str:
    ffmpeg = _ffmpeg()
    if ffmpeg is None:
        raise RuntimeError("cutting clips needs ffmpeg (install it or `pip install imageio-ffmpeg`)")
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-ss", f"{segment.start:.2f}", "-i", path]
    cmd += ["-t", f"{segment.duration:.2f}", "-c:v", "libx264", "-crf", "18", "-c:a", "aac", out_path]
    subprocess.run(cmd, check=True)
    return out_path


def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Cut a long video into main-camera court clips")
    ap.add_argument("video")
    ap.add_argument("--out-dir", default="clips")
    ap.add_argument("--min-len", type=float, default=4.0)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    listing = []
    for i, seg in enumerate(find_court_segments(args.video, min_len_s=args.min_len)):
        clip = str(out / f"clip_{i:03d}.mp4")
        cut(args.video, seg, clip)
        listing.append({"clip": clip, "start": seg.start, "end": seg.end, "corners": seg.corners})
    (out / "segments.json").write_text(json.dumps(listing, indent=2))
    print(f"{len(listing)} clips -> {out}")


if __name__ == "__main__":
    main()
