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

from .court import COURT_LINES, DOUBLES_CORNERS, HALF_DW, HALF_L, HALF_SW, SERVICE_LINE, CourtHomography


def _open_court_points() -> np.ndarray:
    """Court points well away from every painted line, where no line pixels belong.

    Skips the band just beyond the net, where the net itself shows up in the image.
    """
    xs = np.concatenate([np.linspace(-HALF_SW + 0.5, -0.5, 4), np.linspace(0.5, HALF_SW - 0.5, 4)])
    xs = np.concatenate([xs, [-(HALF_SW + HALF_DW) / 2, (HALF_SW + HALF_DW) / 2]])
    ys = np.concatenate(
        [
            np.linspace(-HALF_L + 0.6, -SERVICE_LINE - 0.6, 4),
            np.linspace(-SERVICE_LINE + 0.6, -0.6, 4),
            np.linspace(SERVICE_LINE + 0.6, HALF_L - 0.6, 4),
        ]
    )
    return np.array([(x, y) for x in xs for y in ys])


OPEN_COURT = _open_court_points()


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


def _line_points(n: int = 40) -> np.ndarray:
    ts = np.linspace(0, 1, n)[:, None]
    return np.concatenate([np.array(a) * (1 - ts) + np.array(b) * ts for a, b in COURT_LINES])


LINE_POINTS = _line_points()

# Court y of each horizontal line and |x| of each sideline, for assigning detected
# lines to court lines. The far baseline is often too thin to find in a distant
# broadcast view, so the service lines can stand in for either baseline.
_ROWS = {"near_base": -HALF_L, "near_service": -SERVICE_LINE, "far_service": SERVICE_LINE, "far_base": HALF_L}
_ROW_PAIRS = [
    ("near_base", "far_base"),
    ("near_base", "far_service"),
    ("near_service", "far_base"),
    ("near_service", "far_service"),
]


def _project(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    homo = pts @ m[:, :2].T + m[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return homo[:, :2] / homo[:, 2:3]


def _score_matrix(m: np.ndarray, dist: np.ndarray) -> float:
    """Mean closeness of projected court lines (court -> image matrix `m`) to line pixels.

    Penalized by line pixels in the open court between the lines, so a small
    court fitted into a cluttered patch (stands, logos, a close-up) scores low.
    """
    height, width = dist.shape
    q = _project(m, OPEN_COURT)
    inside = (q[:, 0] >= 0) & (q[:, 0] < width) & (q[:, 1] >= 0) & (q[:, 1] < height)
    clutter = 0.0
    if inside.any():
        p = q[inside].astype(int)
        clutter = float((dist[p[:, 1], p[:, 0]] < 1.5).mean())
    pts = _project(m, LINE_POINTS)
    inside = (pts[:, 0] >= 0) & (pts[:, 0] < width) & (pts[:, 1] >= 0) & (pts[:, 1] < height)
    if not inside.any():
        return -clutter
    p = pts[inside].astype(int)
    total = float(np.clip(1.0 - dist[p[:, 1], p[:, 0]] / 6.0, 0.0, 1.0).sum())
    return total / len(pts) - clutter


def _refine(m: np.ndarray, mask: np.ndarray, dist: np.ndarray, iterations: int = 6) -> np.ndarray:
    """Pulls a rough court fit onto the nearest line pixels (iterative closest point)."""
    height, width = mask.shape
    _, labels = cv2.distanceTransformWithLabels(255 - mask, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(mask)
    # Labels number the line pixels in scan order, which is np.nonzero's order.
    lookup = np.zeros((labels.max() + 1, 2))
    lookup[labels[ys, xs]] = np.stack([xs, ys], axis=1)
    for tol in np.linspace(6.0, 2.5, iterations):
        pts = _project(m, LINE_POINTS)
        ok = np.isfinite(pts).all(axis=1)
        ok[ok] = (pts[ok, 0] >= 0) & (pts[ok, 0] < width) & (pts[ok, 1] >= 0) & (pts[ok, 1] < height)
        p = pts[ok].astype(int)
        near = dist[p[:, 1], p[:, 0]] < tol
        if near.sum() < 12:
            break
        src = LINE_POINTS[ok][near]
        dst = lookup[labels[p[near, 1], p[near, 0]]]
        new, _ = cv2.findHomography(src, dst, 0)
        if new is None:
            break
        m = new
    return m


def follow_court(prev: CourtHomography, frame: np.ndarray, iterations: int = 4) -> tuple[CourtHomography, float]:
    """Moves a court position found in an earlier frame onto this frame's lines.

    For a camera that pans or zooms a little between frames. Returns the new
    position and its score; a low score means the court was lost (a cut, or a
    move too large to follow).
    """
    mask = line_mask(frame)
    dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    m = _refine(prev.court_to_image, mask, dist, iterations)
    corners = _project(m, DOUBLES_CORNERS)
    if not np.isfinite(corners).all():
        return prev, -1.0
    try:
        return CourtHomography(corners), _score_matrix(m, dist)
    except np.linalg.LinAlgError:
        return prev, -1.0


def detect_court(
    frame: np.ndarray, max_lines: int = 7, min_score: float = 0.35
) -> tuple[CourtHomography, float] | None:
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
    best: tuple[np.ndarray, float] | None = None
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
            for near_row, far_row in _ROW_PAIRS:
                y_n, y_f = _ROWS[near_row], _ROWS[far_row]
                for half_w in (HALF_DW, HALF_SW):
                    court_quad = np.array([(-half_w, y_n), (half_w, y_n), (half_w, y_f), (-half_w, y_f)])
                    m = cv2.getPerspectiveTransform(court_quad.astype(np.float32), quad.astype(np.float32))
                    # The whole court must fill a good part of the view.
                    c = _project(m, DOUBLES_CORNERS)
                    if not np.isfinite(c).all() or c[1, 0] - c[0, 0] < 0.25 * width:
                        continue
                    if c[0, 1] - c[3, 1] < 0.15 * height or c[3, 1] > c[0, 1]:
                        continue
                    s = _score_matrix(m, dist)
                    if best is None or s > best[1]:
                        best = (c, s)
    if best is None or best[1] < min_score:
        return None
    rough = cv2.getPerspectiveTransform(DOUBLES_CORNERS.astype(np.float32), best[0].astype(np.float32))
    refined = _refine(rough, mask, dist)
    score = _score_matrix(refined, dist)
    corners, score = (_project(refined, DOUBLES_CORNERS), score) if score >= best[1] else best
    try:
        return CourtHomography(corners), score
    except np.linalg.LinAlgError:
        return None
