"""Rally extraction from broadcast footage.

A broadcast cuts between the main camera behind the baseline and close-ups,
replays, the crowd, commentators and graphics. Only the main camera shows the
whole court, so the video is scanned for stretches where the court lines are
in view ("court shots"), and only those are analyzed.

The main camera may pan and zoom to follow play. It turns about a fixed point,
so any two of its frames are related by a homography, measured from the
background features they share. Every ball detection is mapped into the first
frame of its shot; in those stabilized coordinates the shot looks like a fixed
camera. A broadcast camera looks straight down the court, which leaves its 3D
recovery ill-conditioned, so contacts are read from the image track instead
(`events.detect_events_2d`); line calls and scoring then apply unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .ball import BallTracker, MovingCameraDetector
from .court import COURT_LINES, HALF_DW, HALF_L, CourtCamera, CourtHomography
from .court_detect import detect_court, follow_court
from .events import Rally, find_rallies
from .pipeline import MAX_WIDTH, Options, build_report

NEW_COURT_SCORE = 0.45  # to accept a court found from scratch
KEEP_COURT_SCORE = 0.35  # to keep following a court already found


@dataclass
class CourtShot:
    start: int  # first frame, inclusive
    end: int  # last frame, exclusive
    corners: np.ndarray  # doubles corners at `start`, analysis pixels: NL, NR, FR, FL
    score: float


def _hist(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def _same_scene(a: np.ndarray | None, b: np.ndarray, threshold: float = 0.9) -> bool:
    return a is not None and cv2.compareHist(a, b, cv2.HISTCMP_CORREL) > threshold


class _CourtFinder:
    """Follows the court from frame to frame, searching from scratch when it is lost."""

    def __init__(self) -> None:
        self.hom: CourtHomography | None = None
        self.miss_hist: np.ndarray | None = None

    def __call__(self, frame: np.ndarray) -> tuple[CourtHomography | None, float, bool]:
        """(court or None, score, whether the court was just found from scratch)."""
        if self.hom is not None:
            hom, score = follow_court(self.hom, frame)
            if score >= KEEP_COURT_SCORE:
                self.hom = hom
                return hom, score, False
        hist = _hist(frame)
        # Skip the costly search while a shot without a court simply continues.
        if _same_scene(self.miss_hist, hist, 0.97):
            self.hom = None
            return None, 0.0, False
        found = detect_court(frame, min_score=NEW_COURT_SCORE)
        if found is None:
            self.hom, self.miss_hist = None, hist
            return None, 0.0, False
        self.hom, self.miss_hist = found[0], None
        return found[0], found[1], True


def _analysis_size(cap: cv2.VideoCapture) -> tuple[float, tuple[int, int]]:
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, MAX_WIDTH / src_w)
    return scale, (int(round(src_w * scale)), int(round(src_h * scale)))


def find_court_shots(
    path: str,
    sample_fps: float = 2.0,
    min_len_s: float = 2.0,
    progress: Callable[[float], None] | None = None,
) -> tuple[list[CourtShot], float, int]:
    """Frame ranges where the main camera shows the whole court.

    Samples a few frames a second and follows the court between samples. A
    shot ends where the court is lost or the picture cuts to another scene.
    Returns (shots, fps, frame count).
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps / sample_fps)))
    scale, size = _analysis_size(cap)

    finder = _CourtFinder()
    shots: list[CourtShot] = []
    run: list[tuple[int, float]] = []  # (frame, score) of the court samples in the current shot
    run_corners = None
    prev_hist = None

    def close(end: int) -> None:
        if run and (end - run[0][0]) / fps >= min_len_s:
            shots.append(CourtShot(run[0][0], end, run_corners, float(np.median([s for _, s in run]))))
        run.clear()

    frame_idx = 0
    while True:
        if frame_idx % step:
            if not cap.grab():
                break
            frame_idx += 1
            continue
        ok, raw = cap.read()
        if not ok:
            break
        frame = cv2.resize(raw, size, interpolation=cv2.INTER_AREA) if scale != 1.0 else raw
        hist = _hist(frame)
        hom, score, fresh = finder(frame)
        if hom is None or (fresh and run and not _same_scene(prev_hist, hist)):
            close(run[-1][0] + 1 if run else frame_idx)
        if hom is not None:
            if not run:
                run_corners = hom.image_corners.copy()
            run.append((frame_idx, score))
        prev_hist = hist
        frame_idx += 1
        if progress and total and (frame_idx - 1) % (step * 20) == 0:
            progress(min(1.0, frame_idx / total))
    close(run[-1][0] + 1 if run else frame_idx)
    cap.release()
    return shots, fps, frame_idx


def _play_region(hom: CourtHomography, shape: tuple[int, int]) -> np.ndarray:
    """Where the ball can be in the picture: above the court and its run-off, up to 6 m."""
    x, y = HALF_DW + 3.0, HALF_L + 5.0
    box = np.array([(sx * x, sy * y, z) for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, 6.0)])
    pts = CourtCamera(hom, (shape[1], shape[0])).project(box)
    mask = np.zeros(shape, np.uint8)
    if np.isfinite(pts).all() and np.abs(pts).max() < 1e5:
        cv2.fillConvexPoly(mask, cv2.convexHull(np.round(pts).astype(np.int32)), 255)
    return mask


def _line_pixels(hom: CourtHomography, shape: tuple[int, int]) -> np.ndarray:
    """The painted lines and the net tape, where the slightest misalignment between
    frames shows up as motion."""
    height, width = shape
    thickness = max(3, int(round(7 * width / 1280)))
    mask = np.zeros(shape, np.uint8)
    for a, b in COURT_LINES:
        pa, pb = hom.to_image(np.array([a, b]))
        cv2.line(mask, tuple(np.round(pa).astype(int)), tuple(np.round(pb).astype(int)), 255, thickness)
    post = HALF_DW + 0.914
    net = CourtCamera(hom, (width, height)).project(np.array([(-post, 0, 1.07), (0, 0, 0.914), (post, 0, 1.07)]))
    if np.isfinite(net).all() and np.abs(net).max() < 1e5:
        cv2.polylines(mask, [np.round(net).astype(np.int32)], False, 255, thickness + 2)
    return mask


class _Stabilizer:
    """Maps each frame of a shot onto the shot's first frame.

    Background features of a keyframe are tracked straight into each new frame,
    starting from where the last frame's mapping predicts them, so errors do not
    pile up from frame to frame. When too few of them are still found (the
    camera moved on), the current frame becomes the next keyframe.
    """

    def __init__(self, gray: np.ndarray):
        self.to_ref = np.eye(3)  # current frame -> first frame
        self._keyframe(gray, np.eye(3))

    def _keyframe(self, gray: np.ndarray, to_ref: np.ndarray) -> None:
        self.key_gray, self.key_to_ref = gray, to_ref
        self.key_pts = cv2.goodFeaturesToTrack(gray, maxCorners=500, qualityLevel=0.01, minDistance=10)
        self.key_to_cur = np.eye(3)

    def update(self, gray: np.ndarray) -> np.ndarray | None:
        """Mapping of this frame onto the first frame, or None when the view is lost."""
        if self.key_pts is None or len(self.key_pts) < 20:
            return None
        guess = cv2.perspectiveTransform(self.key_pts, self.key_to_cur).astype(np.float32)
        found, status, _ = cv2.calcOpticalFlowPyrLK(
            self.key_gray,
            gray,
            self.key_pts,
            guess.copy(),
            winSize=(21, 21),
            maxLevel=3,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        )
        ok = status.ravel() == 1
        if ok.sum() < 20:
            return None
        m, inliers = cv2.findHomography(self.key_pts[ok], found[ok], cv2.RANSAC, 1.5)
        if m is None or inliers.sum() < 20:
            return None
        self.key_to_cur = m
        self.to_ref = self.key_to_ref @ np.linalg.inv(m)
        if inliers.sum() < 0.5 * len(self.key_pts):
            self._keyframe(gray, self.to_ref)
        return self.to_ref


def track_shot(
    cap: cv2.VideoCapture,
    shot: CourtShot,
    size: tuple[int, int],
    detector=None,
    progress: Callable[[float], None] | None = None,
) -> tuple[np.ndarray, CourtHomography, float, list[np.ndarray]]:
    """Ball track through one court shot, stabilized to the shot's first frame.

    `detector` defaults to `MovingCameraDetector`; a `tracknet.TrackNetDetector`
    is far more reliable. Returns (track, court homography of the first frame,
    fraction of frames that could be mapped onto the first frame, and for each
    tracked frame the homography mapping it onto the first frame).
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, shot.start)
    ref = CourtHomography(shot.corners)
    if detector is None:
        detector = MovingCameraDetector((size[1], size[0]))
    detector.reset()
    tracker = BallTracker(gate_px=60 * size[0] / 1280)
    n = shot.end - shot.start
    stabilizer = None
    prev_to_ref = None
    followed = 0
    to_refs: list[np.ndarray] = []
    for i in range(n):
        if progress and i % 30 == 0:
            progress(i / max(1, n))
        ok, raw = cap.read()
        if not ok:
            n = i
            break
        frame = cv2.resize(raw, size, interpolation=cv2.INTER_AREA) if raw.shape[1] != size[0] else raw
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if stabilizer is None:
            stabilizer = _Stabilizer(gray)
            to_ref = np.eye(3)
        else:
            to_ref = stabilizer.update(gray)
        if to_ref is None:
            # Lost the picture (a cut the sampling missed); the rest is not this shot.
            n = i
            break
        followed += 1
        to_refs.append(to_ref)
        motion = None if prev_to_ref is None else np.linalg.inv(to_ref) @ prev_to_ref
        prev_to_ref = to_ref
        hom = CourtHomography(CourtHomography._apply(np.linalg.inv(to_ref), ref.image_corners))
        region = _play_region(hom, frame.shape[:2])
        cands = detector.detect(frame, region, _line_pixels(hom, frame.shape[:2]), motion)
        h, w = region.shape
        cands = [c for c in cands if 0 <= c.x < w and 0 <= c.y < h and region[int(c.y), int(c.x)]]
        if cands:
            pts = CourtHomography._apply(to_ref, np.array([[c.x, c.y] for c in cands]))
            for c, (x, y) in zip(cands, pts):
                c.x, c.y = float(x), float(y)
        tracker.update(i, cands)
    return tracker.result(n), ref, followed / max(1, shot.end - shot.start), to_refs


def _shift(rally: Rally, offset: int) -> Rally:
    for e in rally.events:
        e.frame += offset
    for a in rally.arcs:
        a.start += offset
        a.end += offset
    rally.start_frame += offset
    rally.end_frame += offset
    return rally


def analyze_broadcast(
    path: str,
    options: Options | None = None,
    progress: Callable[[float], None] | None = None,
    tracknet_weights: str | None = None,
) -> dict:
    """Finds the court shots in a broadcast, then tracks, judges and scores their rallies."""
    options = options or Options()

    def stage(lo: float, hi: float):
        return (lambda p: progress(lo + (hi - lo) * p)) if progress else None

    shots, fps, n_frames = find_court_shots(path, progress=stage(0.0, 0.3))
    detector = None
    if tracknet_weights:
        from .tracknet import TrackNetDetector

        detector = TrackNetDetector(tracknet_weights)
    cap = cv2.VideoCapture(path)
    scale, size = _analysis_size(cap)
    src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rallies: list[Rally] = []
    shot_out = []
    for k, shot in enumerate(shots):
        track, ref, followed, _ = track_shot(cap, shot, size, detector)
        found = [_shift(r, shot.start) for r in find_rallies(track, ref, fps, size, method="2d")]
        rallies.extend(found)
        shot_out.append(
            {
                "start_t": round(shot.start / fps, 2),
                "end_t": round(shot.end / fps, 2),
                "court_score": round(shot.score, 3),
                "court_followed_pct": round(100 * followed, 1),
                "corners_px": (shot.corners / scale).round(1).tolist(),
                "ball_seen_pct": round(100 * float(np.mean(~np.isnan(track[:, 0]))), 1) if len(track) else 0.0,
                "rallies": len(found),
            }
        )
        if progress:
            progress(0.3 + 0.7 * (k + 1) / max(1, len(shots)))
    cap.release()

    report = build_report(rallies, None, None, fps, n_frames, options)
    report["video"] = {"fps": fps, "frames": n_frames, "width": src_w, "height": src_h, "analysis_scale": scale}
    report["court_shots"] = shot_out
    report["rallies"] = [
        {
            "start_t": round(r.start_frame / fps, 2),
            "end_t": round(r.end_frame / fps, 2),
            "bounces": len(r.bounces),
            "hits": sum(1 for e in r.events if e.kind == "hit"),
        }
        for r in rallies
    ]
    return report


def write_reel(path: str, ranges_s: list[tuple[float, float]], out: str, pad_s: float = 0.5) -> None:
    """Joins the given time ranges of a video into one clip."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for a, b in ranges_s:
        start = max(0, int((a - pad_s) * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(int((b + pad_s) * fps) - start):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
    writer.release()
    cap.release()


def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Extract and analyze the rallies in broadcast tennis footage")
    ap.add_argument("video")
    ap.add_argument("--doubles", action="store_true")
    ap.add_argument("--out", default="-", help="report JSON path")
    ap.add_argument("--reel", help="also write the court shots, joined, to this .mp4")
    ap.add_argument("--tracknet", metavar="WEIGHTS", help="find the ball with TrackNet (needs torch)")
    args = ap.parse_args()
    report = analyze_broadcast(args.video, Options(doubles=args.doubles), tracknet_weights=args.tracknet)
    if args.reel:
        write_reel(args.video, [(s["start_t"], s["end_t"]) for s in report["court_shots"]], args.reel, pad_s=0.0)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out == "-":
        print(text)
    else:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
