"""Automatic court detection from a single frame.

Finds bright line segments, groups them into candidate baselines and sidelines,
and scores every candidate homography by how many painted court lines it lands
on. Works for a fixed camera behind the near baseline with the whole court in
view. The app lets the user tap the four corners when this fails.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import cv2
import numpy as np

from .court import COURT_LINES, HALF_DW, HALF_L, HALF_SW, CourtHomography


@dataclass
class Line:
    theta: float  # normal angle in radians, [0, pi)
    rho: float
    weight: float

    def intersect(self, other: "Line") -> np.ndarray | None:
        a = np.array([[np.cos(self.theta), np.sin(self.theta)], [np.cos(other.theta), np.sin(other.theta)]])
        if abs(np.linalg.det(a)) < 1e-6:
            return None
        return np.linalg.solve(a, np.array([self.rho, other.rho]))

    def x_at(self, y: float) -> float:
        c = np.cos(self.theta)
        return (self.rho - y * np.sin(self.theta)) / c if abs(c) > 1e-6 else np.inf

    def y_at(self, x: float) -> float:
        s = np.sin(self.theta)
        return (self.rho - x * np.cos(self.theta)) / s if abs(s) > 1e-6 else np.inf


def line_mask(frame: np.ndarray) -> np.ndarray:
    """Binary mask of thin bright (white) structures."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k = max(9, (min(frame.shape[:2]) // 40) | 1)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    whiteish = (hsv[..., 1] < 90) & (hsv[..., 2] > 140)
    mask = ((tophat > 30) & whiteish).astype(np.uint8) * 255
    return mask


def _segments_to_lines(segments: np.ndarray) -> list[Line]:
    lines = []
    for x1, y1, x2, y2 in segments.reshape(-1, 4).astype(np.float64):
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        theta = (np.arctan2(dy, dx) + np.pi / 2) % np.pi
        rho = x1 * np.cos(theta) + y1 * np.sin(theta)
        lines.append(Line(theta, rho, length))
    return lines


def _merge(lines: list[Line], angle_tol: float = np.deg2rad(2.5), rho_tol: float = 10.0) -> list[Line]:
    merged: list[Line] = []
    for ln in sorted(lines, key=lambda ln: -ln.weight):
        for m in merged:
            dtheta = abs(ln.theta - m.theta)
            flipped = dtheta > np.pi / 2
            if flipped:
                dtheta = np.pi - dtheta
            rho = -ln.rho if flipped else ln.rho
            if dtheta < angle_tol and abs(rho - m.rho) < rho_tol:
                m.weight += ln.weight
                break
        else:
            merged.append(Line(ln.theta, ln.rho, ln.weight))
    return merged


def _score(h: CourtHomography, dist: np.ndarray, shape: tuple[int, int]) -> float:
    """Mean closeness of projected court lines to detected line pixels."""
    height, width = shape
    total, n = 0.0, 0
    for a, b in COURT_LINES:
        ts = np.linspace(0, 1, 40)[:, None]
        pts = h.to_image(np.array(a) * (1 - ts) + np.array(b) * ts)
        inside = (pts[:, 0] >= 0) & (pts[:, 0] < width) & (pts[:, 1] >= 0) & (pts[:, 1] < height)
        n += len(pts)
        if inside.any():
            p = pts[inside].astype(int)
            d = dist[p[:, 1], p[:, 0]]
            total += float(np.clip(1.0 - d / 6.0, 0.0, 1.0).sum())
    return total / max(n, 1)


def detect_court(frame: np.ndarray, max_lines: int = 7) -> tuple[CourtHomography, float] | None:
    """Returns (homography, score in 0..1) or None when no plausible court was found."""
    height, width = frame.shape[:2]
    mask = line_mask(frame)
    segments = cv2.HoughLinesP(
        mask, 1, np.pi / 360, threshold=40, minLineLength=min(width, height) // 12, maxLineGap=12
    )
    if segments is None:
        return None
    lines = _merge(_segments_to_lines(segments))

    # Normal angle near 90 degrees means a near-horizontal line.
    horizontal = [ln for ln in lines if abs(ln.theta - np.pi / 2) < np.deg2rad(20)]
    steep = [ln for ln in lines if abs(ln.theta - np.pi / 2) >= np.deg2rad(20)]
    horizontal = sorted(horizontal, key=lambda ln: -ln.weight)[:max_lines]
    steep = sorted(steep, key=lambda ln: -ln.weight)[:max_lines]
    if len(horizontal) < 2 or len(steep) < 2:
        return None

    dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    best: tuple[CourtHomography, float] | None = None
    cx = width / 2
    for h_pair in itertools.combinations(horizontal, 2):
        far, near = sorted(h_pair, key=lambda ln: ln.y_at(cx))
        for s_pair in itertools.combinations(steep, 2):
            left, right = sorted(s_pair, key=lambda ln: ln.x_at(height / 2))
            pts = [near.intersect(left), near.intersect(right), far.intersect(right), far.intersect(left)]
            if any(p is None for p in pts):
                continue
            quad = np.array(pts)
            # Far edge must be shorter than the near edge and above it.
            if not (quad[2, 0] - quad[3, 0] < quad[1, 0] - quad[0, 0] and quad[3, 1] < quad[0, 1]):
                continue
            if np.abs(quad).max() > 4 * max(width, height):
                continue
            for half_w in (HALF_DW, HALF_SW):
                # Corners of the chosen sidelines, expanded to doubles corners.
                court_quad = np.array([(-half_w, -HALF_L), (half_w, -HALF_L), (half_w, HALF_L), (-half_w, HALF_L)])
                m = cv2.getPerspectiveTransform(court_quad.astype(np.float32), quad.astype(np.float32))
                doubles = np.array([(-HALF_DW, -HALF_L), (HALF_DW, -HALF_L), (HALF_DW, HALF_L), (-HALF_DW, HALF_L)])
                corners = CourtHomography._apply(m, doubles)
                try:
                    hom = CourtHomography(corners)
                except np.linalg.LinAlgError:
                    continue
                s = _score(hom, dist, (height, width))
                if best is None or s > best[1]:
                    best = (hom, s)
    if best is None or best[1] < 0.35:
        return None
    return best
