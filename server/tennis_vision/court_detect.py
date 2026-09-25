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
    return _line_images(frame)[0]


def _line_images(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(binary line mask, top-hat brightness) of a frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k = max(9, (min(frame.shape[:2]) // 40) | 1)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Distant lines are thin and pick up the court's tint, so stay permissive here;
    # the model fit below rejects whatever is not court geometry.
    whiteish = (hsv[..., 1] < 125) & (hsv[..., 2] > 130)
    mask = ((tophat > 25) & whiteish).astype(np.uint8) * 255
    return mask, tophat


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


_TS = np.linspace(0, 1, 40)[:, None]
_LINE_SAMPLES = np.concatenate([np.array(a) * (1 - _TS) + np.array(b) * _TS for a, b in COURT_LINES])


def _score(h: CourtHomography, dist: np.ndarray, shape: tuple[int, int]) -> float:
    """Mean closeness of projected court lines to detected line pixels."""
    height, width = shape
    pts = h.to_image(_LINE_SAMPLES)
    inside = (pts[:, 0] >= 0) & (pts[:, 0] < width) & (pts[:, 1] >= 0) & (pts[:, 1] < height)
    p = pts[inside].astype(int)
    d = dist[p[:, 1], p[:, 0]]
    return float(np.clip(1.0 - d / 6.0, 0.0, 1.0).sum()) / len(pts)


def detect_court(frame: np.ndarray, max_lines: int = 10) -> tuple[CourtHomography, float] | None:
    """Returns (homography, score in 0..1) or None when no plausible court was found."""
    height, width = frame.shape[:2]
    mask, tophat = _line_images(frame)
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
            if not _plausible(quad, width, height):
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
    refined = _refine(best[0], mask, tophat)
    if refined is not None and _plausible(refined.image_corners, width, height):
        s = _score(refined, dist, (height, width))
        if s >= best[1] - 0.01:
            best = (refined, s)
    return best


def _plausible(quad: np.ndarray, width: int, height: int) -> bool:
    """Whether a near-left, near-right, far-right, far-left quad can be a court seen from behind a baseline."""
    # Far edge must be shorter than the near edge and above it.
    if not (quad[2, 0] - quad[3, 0] < quad[1, 0] - quad[0, 0] and quad[3, 1] < quad[0, 1]):
        return False
    if np.abs(quad).max() > 4 * max(width, height):
        return False
    # A collapsed quad projects every court line onto one detected line and
    # scores perfectly, so require a court of plausible size.
    x, y = quad[:, 0], quad[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    if area < 0.03 * width * height or min(quad[0, 1] - quad[3, 1], quad[1, 1] - quad[2, 1]) < 0.1 * height:
        return False
    # Perspective shrinks the far baseline, but never to a point: a court
    # is at most about 4x deeper than wide from any sensible camera.
    near_w, far_w = quad[1, 0] - quad[0, 0], quad[2, 0] - quad[3, 0]
    return 0.2 * near_w < far_w <= near_w


_DOUBLES = np.array([(-HALF_DW, -HALF_L), (HALF_DW, -HALF_L), (HALF_DW, HALF_L), (-HALF_DW, HALF_L)])
_OFFSETS = np.arange(-6, 7)


def _nearest_run_center(row: np.ndarray, brightness: np.ndarray) -> float:
    """Offset of the center of the run of line pixels closest to the probe's middle.

    Taking only one run keeps a neighboring line or a player's white kit from
    dragging the sample off its own line.
    """
    idx = np.flatnonzero(row)
    if len(idx) == 0:
        return np.nan
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    mid = len(row) // 2
    run = min(runs, key=lambda r: 0 if r[0] <= mid <= r[-1] else min(abs(r[0] - mid), abs(r[-1] - mid)))
    if len(run) > 8:  # a blob, not a line
        return np.nan
    # Brightness-weighted, including the dim anti-aliased pixel on each edge.
    span = np.arange(max(run[0] - 1, 0), min(run[-1] + 2, len(row)))
    w = brightness[span].astype(np.float64)
    return float((_OFFSETS[span] * w).sum() / w.sum())


def _refine(
    hom: CourtHomography, mask: np.ndarray, brightness: np.ndarray, iterations: int = 6
) -> CourtHomography | None:
    """Pull the court model onto the centers of the painted lines.

    The candidate search only uses Hough line fits, which sit a pixel or more off
    the true line centers. Here every sample along the projected court lines
    looks across the line for the painted pixels and moves to their centroid;
    a homography fitted to those points is repeated a few times.
    """
    height, width = mask.shape
    for _ in range(iterations):
        src, dst = [], []
        for a, b in COURT_LINES:
            ends = hom.to_image(np.array([a, b]))
            tangent = ends[1] - ends[0]
            length = np.linalg.norm(tangent)
            if length < 1:
                continue
            normal = np.array([-tangent[1], tangent[0]]) / length
            court_pts = np.array(a) * (1 - _TS) + np.array(b) * _TS
            img_pts = hom.to_image(court_pts)
            probe = img_pts[:, None, :] + _OFFSETS[None, :, None] * normal  # (n, k, 2)
            xy = probe.round().astype(int)
            valid = (xy[..., 0] >= 0) & (xy[..., 0] < width) & (xy[..., 1] >= 0) & (xy[..., 1] < height)
            w = np.zeros(valid.shape)
            w[valid] = mask[xy[..., 1][valid], xy[..., 0][valid]] > 0
            b = np.zeros(valid.shape)
            b[valid] = brightness[xy[..., 1][valid], xy[..., 0][valid]]
            shift = np.array([_nearest_run_center(row, br) for row, br in zip(w, b)])
            ok = ~np.isnan(shift)
            if not ok.any():
                continue
            shift = shift[ok]
            src.append(court_pts[ok])
            dst.append(img_pts[ok] + shift[:, None] * normal)
        if not src:
            return None
        src, dst = np.concatenate(src), np.concatenate(dst)
        if len(src) < 20:
            return None
        m, _ = cv2.findHomography(src.astype(np.float32), dst.astype(np.float32), cv2.RANSAC, 5.0)
        if m is None:
            return None
        try:
            hom = CourtHomography(CourtHomography._apply(m, _DOUBLES))
        except np.linalg.LinAlgError:
            return None
    return hom
