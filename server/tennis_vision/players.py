"""Where the two players stand: person detection, mapped onto the court.

A person detector (YOLO through `ultralytics`, an optional dependency) runs every
few frames. Each detection's feet (bottom center of its box) map through the
court homography to meters, and the biggest person on each half of the court is
taken as that half's player. Ball kids, the umpire and people behind the fence
fall outside the play area or are smaller, so they drop out.

The positions feed shot classification (which side of the body the ball was
struck on) and movement stats (distance covered, where each player stood).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .court import HALF_DW, HALF_L, CourtHomography, Side

PLAYER_STRIDE = 3  # run detection on every 3rd frame
MAX_SPEED_MS = 9.0  # faster than any player sprints; larger jumps are detection swaps
PLAY_AREA_X = HALF_DW + 3.5  # meters from the center line a player can stand
PLAY_AREA_Y = HALF_L + 7.0  # meters from the net


class PersonDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        """Person boxes in frame pixels: x1, y1, x2, y2, confidence."""
        ...


class YoloPersonDetector:
    """People found by a YOLO model. The weights download on first use."""

    def __init__(self, model: str | None = None, imgsz: int = 960, conf: float = 0.25):
        from ultralytics import YOLO  # optional dependency: pip install -e '.[players]'

        self.model = YOLO(model or os.environ.get("TENNIS_PLAYER_MODEL", "yolo11n.pt"))
        self.imgsz = imgsz
        self.conf = conf

    def detect(self, frame: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        res = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, classes=[0], verbose=False)[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        return [(*map(float, b), float(c)) for b, c in zip(xyxy, conf, strict=True)]


def default_detector() -> PersonDetector | None:
    """The YOLO detector when `ultralytics` is installed and not switched off, else None."""
    if os.environ.get("TENNIS_PLAYERS", "1") == "0":
        return None
    try:
        return YoloPersonDetector()
    except ImportError:
        return None


@dataclass
class PlayerTrack:
    """One player's feet on the court over time (sampled frames only)."""

    frames: list[int] = field(default_factory=list)
    xy: list[tuple[float, float]] = field(default_factory=list)
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)

    def at(self, frame: float, max_gap: int = 2 * PLAYER_STRIDE + 1) -> tuple[float, float] | None:
        """Position at `frame`, interpolated between samples no more than `max_gap` frames apart."""
        if not self.frames:
            return None
        f = np.asarray(self.frames)
        i = int(np.searchsorted(f, frame))
        if i < len(f) and f[i] == frame:
            return self.xy[i]
        lo, hi = i - 1, i
        if lo >= 0 and hi < len(f) and f[hi] - f[lo] <= max_gap:
            w = (frame - f[lo]) / (f[hi] - f[lo])
            a, b = np.asarray(self.xy[lo]), np.asarray(self.xy[hi])
            return tuple(float(v) for v in a + w * (b - a))
        near = [k for k in (lo, hi) if 0 <= k < len(f) and abs(f[k] - frame) <= max_gap]
        if not near:
            return None
        k = min(near, key=lambda k: abs(f[k] - frame))
        return self.xy[k]

    def distance_m(self, fps: float) -> float:
        """Distance covered, skipping jumps faster than a person can run."""
        total = 0.0
        for k in range(1, len(self.frames)):
            dt = (self.frames[k] - self.frames[k - 1]) / fps
            step = float(np.hypot(*(np.asarray(self.xy[k]) - np.asarray(self.xy[k - 1]))))
            if dt <= 1.0 and step <= MAX_SPEED_MS * dt:
                total += step
        return total


class PlayerTracker:
    def __init__(self, homography: CourtHomography):
        self.homography = homography
        self.tracks = {Side.NEAR: PlayerTrack(), Side.FAR: PlayerTrack()}

    def update(self, frame_idx: int, boxes: list[tuple[float, float, float, float, float]]) -> None:
        if not boxes:
            return
        b = np.asarray(boxes, dtype=np.float64)
        feet = np.stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]], axis=1)
        court = self.homography.to_court(feet)
        height = b[:, 3] - b[:, 1]
        for side in (Side.NEAR, Side.FAR):
            on_side = court[:, 1] < 0 if side is Side.NEAR else court[:, 1] >= 0
            ok = on_side & (np.abs(court[:, 0]) < PLAY_AREA_X) & (np.abs(court[:, 1]) < PLAY_AREA_Y)
            if not ok.any():
                continue
            track = self.tracks[side]
            idx = np.where(ok)[0]
            if track.frames and frame_idx - track.frames[-1] <= 4 * PLAYER_STRIDE:
                # Prefer whoever is closest to where the player just was, then the biggest.
                last = np.asarray(track.xy[-1])
                dist = np.linalg.norm(court[idx] - last, axis=1)
                close = idx[dist < 2.5]
                k = close[np.argmax(height[close])] if len(close) else idx[np.argmax(height[idx])]
            else:
                k = idx[np.argmax(height[idx])]
            track.frames.append(frame_idx)
            track.xy.append((float(court[k, 0]), float(court[k, 1])))
            track.boxes.append(tuple(float(v) for v in b[k, :4]))


def movement_stats(track: PlayerTrack, fps: float) -> dict:
    if not track.frames:
        return {"distance_m": None, "positions": []}
    # A thinned position list for the app's court heatmap: about two per second.
    step = max(1, round(fps / 2 / PLAYER_STRIDE))
    return {
        "distance_m": round(track.distance_m(fps), 1),
        "positions": [[round(x, 2), round(y, 2)] for x, y in track.xy[::step]],
    }
