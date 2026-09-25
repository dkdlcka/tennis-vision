"""Ball detection and tracking for a fixed camera.

The classical detector here uses motion (background subtraction plus
three-frame differencing) and the ball's yellow-green color. It is a baseline
that runs anywhere; `BallDetector` is the seam for dropping in a learned
detector such as TrackNet once we have labeled footage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np


@dataclass
class Candidate:
    x: float
    y: float
    score: float


class BallDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Candidate]: ...


class MotionColorDetector:
    """Finds small, moving, ball-colored blobs."""

    def __init__(self, frame_size: tuple[int, int], court_mask: np.ndarray | None = None):
        height, width = frame_size
        scale = width / 1280
        self.min_area = max(2.0, 3 * scale * scale)
        self.max_area = 400 * scale * scale
        self.court_mask = court_mask
        self.bg = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=24, detectShadows=False)
        self.prev: list[np.ndarray] = []

    def detect(self, frame: np.ndarray) -> list[Candidate]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg = self.bg.apply(frame)
        self.prev.append(gray)
        if len(self.prev) > 3:
            self.prev.pop(0)
        if len(self.prev) < 3:
            return []
        d1 = cv2.absdiff(self.prev[2], self.prev[1])
        d2 = cv2.absdiff(self.prev[1], self.prev[0])
        # Pixels that differ from the background now and changed recently. Requiring
        # the background test drops the ghost the ball leaves where it just was.
        raw = ((fg > 0) & ((d1 > 18) | (d2 > 18))).astype(np.uint8) * 255
        motion = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        motion = cv2.dilate(motion, np.ones((3, 3), np.uint8))
        if self.court_mask is not None:
            motion &= self.court_mask

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ball_color = cv2.inRange(hsv, (22, 60, 90), (50, 255, 255))

        background = self.bg.getBackgroundImage()
        bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY) if background is not None else None

        n, labels, stats, centroids = cv2.connectedComponentsWithStats(motion, connectivity=8)
        out = []
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if not (self.min_area <= area <= self.max_area):
                continue
            w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            # Motion blur stretches the ball, but not into a long line.
            elongation = max(w, h) / max(1, min(w, h))
            if elongation > 6:
                continue
            ys, xs = slice(stats[i, 1], stats[i, 1] + h), slice(stats[i, 0], stats[i, 0] + w)
            region = labels[ys, xs] == i
            # A ball is brighter than the court behind it; shadows are darker.
            if bg_gray is not None and gray[ys, xs][region].mean() < bg_gray[ys, xs][region].mean():
                continue
            color = ball_color[ys, xs][region]
            color_frac = float((color > 0).mean()) if color.size else 0.0
            score = 0.3 + 0.7 * color_frac - 0.05 * (elongation - 1)
            # Center from the unfiltered pixels: the morphology above shifts blobs slightly.
            yy, xx = np.nonzero(region & (raw[ys, xs] > 0) & (d1[ys, xs] > 18))
            if len(xx):
                cx, cy = stats[i, 0] + xx.mean(), stats[i, 1] + yy.mean()
            else:
                cx, cy = centroids[i]
            out.append(Candidate(float(cx), float(cy), score))
        return out


@dataclass
class _Track:
    points: dict[int, tuple[float, float]] = field(default_factory=dict)
    last_frame: int = -1

    def predict(self, frame: int) -> np.ndarray:
        frames = sorted(self.points)[-3:]
        p = np.array([self.points[f] for f in frames])
        if len(p) == 1:
            return p[-1]
        v = (p[-1] - p[-2]) / (frames[-1] - frames[-2])
        return p[-1] + v * (frame - frames[-1])


class BallTracker:
    """Links per-frame candidates into one ball trajectory.

    Keeps several tentative tracks, extends each with the candidate nearest its
    prediction, and returns the positions from tracks that lasted long enough to
    be the ball rather than noise.
    """

    def __init__(self, gate_px: float = 60.0, max_gap: int = 6, min_length: int = 6):
        self.gate_px = gate_px
        self.max_gap = max_gap
        self.min_length = min_length
        self.tracks: list[_Track] = []
        self.finished: list[_Track] = []

    def update(self, frame: int, candidates: list[Candidate]) -> None:
        free = list(candidates)
        for tr in sorted(self.tracks, key=lambda t: -len(t.points)):
            if not free:
                break
            pred = tr.predict(frame)
            gap = frame - tr.last_frame
            gate = self.gate_px * (1 + 0.5 * (gap - 1))
            last = np.array(tr.points[tr.last_frame])
            step = np.linalg.norm(pred - last)
            # Favor ball-colored blobs so the tracker does not wander onto the shadow,
            # and skip the ghost a fast ball can leave where it was last frame.
            dists = [
                np.inf
                if step > 2 and np.hypot(c.x - last[0], c.y - last[1]) < 0.3 * step
                else np.hypot(c.x - pred[0], c.y - pred[1]) + 40 * (1 - c.score)
                for c in free
            ]
            j = int(np.argmin(dists))
            if dists[j] < gate:
                c = free.pop(j)
                tr.points[frame] = (c.x, c.y)
                tr.last_frame = frame
        for c in free:
            if c.score > 0.5:
                self.tracks.append(_Track({frame: (c.x, c.y)}, frame))
        alive = []
        for tr in self.tracks:
            if frame - tr.last_frame > self.max_gap:
                self.finished.append(tr)
            else:
                alive.append(tr)
        self.tracks = alive

    def result(self, n_frames: int, interpolate: bool = False) -> np.ndarray:
        """(n_frames, 2) array of ball pixels, NaN where the ball was not seen."""
        out = np.full((n_frames, 2), np.nan)
        tracks = [t for t in self.finished + self.tracks if len(t.points) >= self.min_length]
        # Longer tracks win when two claim the same frame.
        for tr in sorted(tracks, key=lambda t: len(t.points)):
            for f, p in tr.points.items():
                if f < n_frames:
                    out[f] = p
        return interpolate_gaps(out, self.max_gap) if interpolate else out


def interpolate_gaps(track: np.ndarray, max_gap: int) -> np.ndarray:
    out = track.copy()
    seen = np.where(~np.isnan(out[:, 0]))[0]
    for a, b in zip(seen[:-1], seen[1:]):
        if 1 < b - a <= max_gap + 1:
            for k in range(a + 1, b):
                t = (k - a) / (b - a)
                out[k] = out[a] * (1 - t) + out[b] * t
    return out
