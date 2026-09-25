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
    min_score: float = 0.6,
) -> list[Segment]:
    """Runs of sampled frames where the court appears at the dominant position.

    The main camera's court position is taken as the most common one among all
    detections, so a replay from another angle or a zoomed shot is left out
    even when a court is visible in it.
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(step_s * fps)))
    samples: list[tuple[float, np.ndarray | None]] = []
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
        scale = min(1.0, SAMPLE_WIDTH / frame.shape[1])
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
        found = detect_court(small)
        corners = found[0].image_corners / scale if found and found[1] >= min_score else None
        samples.append((f / fps, corners))
    cap.release()

    detected = [c for _, c in samples if c is not None]
    if not detected:
        return []
    width = max(float(np.ptp(np.concatenate(detected)[:, 0])), 1.0)
    tol = max_shift_frac * width

    # Dominant position: the detection with the most others within tolerance.
    stack = np.array(detected)
    dists = np.abs(stack[:, None] - stack[None]).max(axis=(2, 3))
    ref = stack[int(np.argmax((dists < tol).sum(axis=1)))]

    segments: list[Segment] = []
    run: list[tuple[float, np.ndarray]] = []

    def close_run() -> None:
        if run and run[-1][0] - run[0][0] + step_s >= min_len_s:
            corners = np.median(np.array([c for _, c in run]), axis=0)
            segments.append(Segment(run[0][0], run[-1][0] + step_s, corners.round(1).tolist()))

    for t, c in samples:
        if c is not None and np.abs(c - ref).max() < tol:
            run.append((t, c))
        else:
            close_run()
            run = []
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
