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


class StreakDetector:
    """Finds the ball as a small patch brighter than it was a frame before and a frame after.

    Broadcast and compressed 25-30 fps video blurs a fast ball into a faint
    grayish streak whose color is unreliable, but it is still lighter than the
    court it crosses. Comparing the middle of three frames with both neighbors
    (signed, so dark shadows and ghosts drop out) isolates it; painted lines and
    text that flicker with small alignment errors and moving players are masked.

    `detect(frame)` returns the candidates of the previous frame, since it
    needs the next frame to decide.
    """

    def __init__(self, frame_size: tuple[int, int], static: np.ndarray | None = None, threshold: int = 12):
        height, width = frame_size
        s = width / 1280
        self.min_area = max(2.0, 2 * s * s)
        self.max_area = 150 * s * s
        self.player_area = 400 * s * s
        self.player_margin = max(3, int(12 * s)) | 1
        self.threshold = threshold
        self.static = static
        self.bg = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=24, detectShadows=False)
        self.prev: list[np.ndarray] = []
        self.backgrounds: list[np.ndarray | None] = []

    def detect(self, frame: np.ndarray) -> list[Candidate]:
        self.bg.apply(frame)
        background = self.bg.getBackgroundImage()
        self.prev.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16))
        self.backgrounds.append(
            cv2.cvtColor(background, cv2.COLOR_BGR2GRAY).astype(np.int16) if background is not None else None
        )
        if len(self.prev) > 3:
            self.prev.pop(0)
            self.backgrounds.pop(0)
        if len(self.prev) < 3:
            return []
        before, cur, after = self.prev
        # Background as of the middle frame, for the whole ball rather than the
        # part that did not overlap its neighbors.
        bg = self.backgrounds[1]
        up = np.minimum(cur - before, cur - after)
        moving = np.maximum(np.abs(cur - before), np.abs(cur - after)) > self.threshold
        # Large moving regions are players (and their rackets); the ball near them is lost anyway.
        n, labels, stats, _ = cv2.connectedComponentsWithStats(moving.astype(np.uint8), connectivity=8)
        big = np.isin(labels, np.flatnonzero(stats[:, cv2.CC_STAT_AREA] > self.player_area)[1:])
        players = cv2.dilate(big.astype(np.uint8), np.ones((self.player_margin, self.player_margin), np.uint8))
        mask = (up > self.threshold) & (players == 0)
        if self.static is not None:
            mask &= self.static == 0

        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        out = []
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if not (self.min_area <= area <= self.max_area):
                continue
            w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            elongation = max(w, h) / max(1, min(w, h))
            if elongation > 8:
                continue
            ys, xs = slice(stats[i, 1], stats[i, 1] + h), slice(stats[i, 0], stats[i, 0] + w)
            region = labels[ys, xs] == i
            contrast = float(up[ys, xs][region].mean())
            cx, cy = self._center(cur, bg, stats[i], region, centroids[i])
            score = min(1.0, 0.3 + contrast / 50) - 0.03 * (elongation - 1)
            out.append(Candidate(cx, cy, score))
        return out

    def _center(self, cur, bg, stat, seed, fallback) -> tuple[float, float]:
        """Brightness-weighted center of the ball's pixels that differ from the background."""
        if bg is None:
            return float(fallback[0]), float(fallback[1])
        x0, y0, w, h = stat[0], stat[1], stat[2], stat[3]
        pad = max(w, h)
        ya, yb = max(0, y0 - pad), min(cur.shape[0], y0 + h + pad)
        xa, xb = max(0, x0 - pad), min(cur.shape[1], x0 + w + pad)
        lift = cur[ya:yb, xa:xb] - bg[ya:yb, xa:xb]
        blob = (lift > self.threshold).astype(np.uint8)
        n, labels = cv2.connectedComponents(blob, connectivity=8)
        seed_labels = labels[y0 - ya : y0 - ya + h, x0 - xa : x0 - xa + w][seed]
        seed_labels = seed_labels[seed_labels > 0]
        if len(seed_labels) == 0:
            return float(fallback[0]), float(fallback[1])
        region = labels == np.bincount(seed_labels).argmax()
        if region.sum() > 4 * self.max_area:
            return float(fallback[0]), float(fallback[1])
        yy, xx = np.nonzero(region)
        weight = lift[region].astype(np.float64)
        return xa + float((xx * weight).sum() / weight.sum()), ya + float((yy * weight).sum() / weight.sum())


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

    def __init__(self, gate_px: float = 60.0, max_gap: int = 6, min_length: int = 6, min_travel_px: float = 30.0):
        self.gate_px = gate_px
        self.min_travel_px = min_travel_px
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
        scored = [(self._smooth_steps(t), t) for t in self.finished + self.tracks if len(t.points) >= self.min_length]
        tracks = [(ok, t) for ok, t in scored if self._is_ball(t, ok)]
        # There is one ball: the track that follows a smooth path the longest owns
        # its time span, and weaker tracks keep only what lies outside it.
        taken: list[tuple[int, int]] = []
        for ok, tr in sorted(tracks, key=lambda x: -sum(x[0])):
            frames = [f for f in sorted(tr.points) if f < n_frames and not any(a <= f <= b for a, b in taken)]
            if len(frames) < self.min_length:
                continue
            for f in frames:
                out[f] = tr.points[f]
            taken.append((frames[0], frames[-1]))
        return interpolate_gaps(out, self.max_gap) if interpolate else out

    @staticmethod
    def _smooth_steps(tr: _Track) -> list[bool]:
        """For each run of three consecutive frames, whether the track moves at a steady velocity."""
        frames = np.array(sorted(tr.points))
        p = np.array([tr.points[f] for f in frames])
        return [
            bool(np.linalg.norm(p[i + 2] - 2 * p[i + 1] + p[i]) < max(4.0, 0.5 * np.linalg.norm(p[i + 2] - p[i + 1])))
            for i in range(len(p) - 2)
            if frames[i + 2] - frames[i] == 2
        ]

    def _is_ball(self, tr: _Track, ok: list[bool]) -> bool:
        """A ball in play goes somewhere along a smooth path, passing the steady-velocity
        check everywhere but at its contacts; flicker in the crowd, on a sign or on a
        TV graphic stays put or hops around at random."""
        p = np.array(list(tr.points.values()))
        if np.ptp(p, axis=0).max() < self.min_travel_px:
            return False
        # Too few consecutive frames to judge means the track hopped between gaps.
        if len(ok) < max(3, 0.3 * len(p)):
            return False
        return float(np.mean(ok)) >= 0.6


def interpolate_gaps(track: np.ndarray, max_gap: int) -> np.ndarray:
    out = track.copy()
    seen = np.where(~np.isnan(out[:, 0]))[0]
    for a, b in zip(seen[:-1], seen[1:]):
        if 1 < b - a <= max_gap + 1:
            for k in range(a + 1, b):
                t = (k - a) / (b - a)
                out[k] = out[a] * (1 - t) + out[b] * t
    return out
