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

        background = self.bg.getBackgroundImage()
        bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY) if background is not None else None
        return _blobs(frame, gray, bg_gray, raw, motion, d1, self.min_area, self.max_area)


class MovingCameraDetector:
    """`MotionColorDetector` for a camera that pans and zooms, like a broadcast.

    A camera that only turns and zooms maps whole frames onto each other with a
    homography, measured from background features tracked between frames. The
    recent frames are warped onto the current one: the frame from a few frames
    back stands in for the background, the last two show what just moved.
    """

    def __init__(self, frame_size: tuple[int, int], depth: int = 6):
        height, width = frame_size
        scale = width / 1280
        self.min_area = max(2.0, 3 * scale * scale)
        self.max_area = 400 * scale * scale
        self.small = width < 1000
        self.depth = depth
        self.history: list[tuple[np.ndarray, np.ndarray]] = []  # (gray, maps it onto the latest frame)

    def reset(self) -> None:
        self.history = []

    def detect(
        self,
        frame: np.ndarray,
        region: np.ndarray | None = None,
        ignore: np.ndarray | None = None,
        motion: np.ndarray | None = None,
    ) -> list[Candidate]:
        """Candidates in this frame. `region` limits where to look; `ignore` masks out
        pixels (such as painted lines) that misalignment would light up. `motion`
        maps the previous frame onto this one when the caller has measured it."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.history and motion is None:
            motion = frame_motion(self.history[-1][0], gray)
            if motion is None:  # a cut, or too little texture to follow the camera
                self.reset()
        self.history = [(g, motion @ m) for g, m in self.history]
        out: list[Candidate] = []
        if len(self.history) == self.depth:
            out = self._detect(frame, gray, region, ignore)
        self.history = (self.history + [(gray, np.eye(3))])[-self.depth :]
        return out

    def _detect(self, frame, gray, region, ignore) -> list[Candidate]:
        size = (gray.shape[1], gray.shape[0])
        valid = np.full(gray.shape, 255, np.uint8)
        warped = []
        for g, m in (self.history[0], self.history[-2], self.history[-1]):
            warped.append(cv2.warpPerspective(g, m, size, flags=cv2.INTER_LINEAR, borderValue=0))
            valid &= cv2.warpPerspective(np.full_like(g, 255), m, size, flags=cv2.INTER_NEAREST, borderValue=0)
        valid = cv2.erode(valid, np.ones((5, 5), np.uint8))
        if region is not None:
            valid &= region
        if ignore is not None:
            valid &= ~ignore
        background, before_last, last = warped
        d_bg = cv2.absdiff(gray, background)
        d1 = cv2.absdiff(gray, last)
        d2 = cv2.absdiff(last, before_last)
        # Differs from the background now and changed recently; the ghost the
        # ball leaves in the background frame fails the brightness test later.
        raw = ((d_bg > 18) & ((d1 > 18) | (d2 > 18)) & (valid > 0)).astype(np.uint8) * 255
        # In small frames a distant ball is only a pixel or two wide, which an
        # opening would erase; the painted lines are masked out instead.
        motion = raw if self.small else cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        motion = cv2.dilate(motion, np.ones((3, 3), np.uint8))
        return _blobs(frame, gray, background, raw, motion, d1, self.min_area, self.max_area)


def frame_motion(prev: np.ndarray, cur: np.ndarray) -> np.ndarray | None:
    """Homography mapping `prev` onto `cur`, from tracked background features; None on a cut."""
    pts = cv2.goodFeaturesToTrack(prev, maxCorners=400, qualityLevel=0.01, minDistance=8)
    if pts is None or len(pts) < 20:
        return None
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, winSize=(21, 21), maxLevel=3)
    ok = status.ravel() == 1
    if ok.sum() < 20:
        return None
    m, inliers = cv2.findHomography(pts[ok], nxt[ok], cv2.RANSAC, 2.0)
    if m is None or inliers.sum() < 0.5 * ok.sum():
        return None
    return m


def _blobs(
    frame: np.ndarray,
    gray: np.ndarray,
    bg_gray: np.ndarray | None,
    raw: np.ndarray,
    motion: np.ndarray,
    d1: np.ndarray,
    min_area: float,
    max_area: float,
) -> list[Candidate]:
    """Scores moving blobs by size, shape, brightness and ball color."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ball_color = cv2.inRange(hsv, (22, 60, 90), (50, 255, 255))
    # Broadcast cameras often blow the ball out to a pale, near-white yellow-green.
    ball_color |= cv2.inRange(hsv, (30, 15, 245), (70, 110, 255))

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(motion, connectivity=8)
    out = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (min_area <= area <= max_area):
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
