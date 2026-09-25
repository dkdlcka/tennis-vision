"""Court model, homography, and in/out geometry.

Court coordinates are in meters with the origin at the center of the net:
  x runs across the court (+x is the right side as seen from the camera),
  y runs along the court (-y is the near half, +y the far half).
The camera is assumed to sit behind the near baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

COURT_LENGTH = 23.77
DOUBLES_WIDTH = 10.97
SINGLES_WIDTH = 8.23
SERVICE_LINE = 6.40  # distance from the net
BALL_RADIUS = 0.033

HALF_L = COURT_LENGTH / 2
HALF_DW = DOUBLES_WIDTH / 2
HALF_SW = SINGLES_WIDTH / 2


class Side(str, Enum):
    NEAR = "near"
    FAR = "far"

    @property
    def other(self) -> "Side":
        return Side.FAR if self is Side.NEAR else Side.NEAR


class ServeBox(str, Enum):
    DEUCE = "deuce"
    AD = "ad"


# Every painted line as a segment in court coordinates.
COURT_LINES: list[tuple[tuple[float, float], tuple[float, float]]] = [
    ((-HALF_DW, -HALF_L), (HALF_DW, -HALF_L)),  # near baseline
    ((-HALF_DW, HALF_L), (HALF_DW, HALF_L)),  # far baseline
    ((-HALF_DW, -HALF_L), (-HALF_DW, HALF_L)),  # left doubles sideline
    ((HALF_DW, -HALF_L), (HALF_DW, HALF_L)),  # right doubles sideline
    ((-HALF_SW, -HALF_L), (-HALF_SW, HALF_L)),  # left singles sideline
    ((HALF_SW, -HALF_L), (HALF_SW, HALF_L)),  # right singles sideline
    ((-HALF_SW, -SERVICE_LINE), (HALF_SW, -SERVICE_LINE)),  # near service line
    ((-HALF_SW, SERVICE_LINE), (HALF_SW, SERVICE_LINE)),  # far service line
    ((0.0, -SERVICE_LINE), (0.0, SERVICE_LINE)),  # center service line
]

# Outer corners of the doubles court, ordered near-left, near-right, far-right, far-left.
DOUBLES_CORNERS = np.array(
    [(-HALF_DW, -HALF_L), (HALF_DW, -HALF_L), (HALF_DW, HALF_L), (-HALF_DW, HALF_L)],
    dtype=np.float64,
)


@dataclass(frozen=True)
class Rect:
    x0: float
    x1: float
    y0: float
    y1: float

    def margin(self, x: float, y: float) -> float:
        """Signed distance in meters from (x, y) to the rectangle edge.

        Positive means inside, negative means outside. Lines belong to the
        area they enclose, so a ball landing on the outer edge of a line is in.
        """
        inside = min(x - self.x0, self.x1 - x, y - self.y0, self.y1 - y)
        if inside >= 0:
            return inside
        dx = max(self.x0 - x, 0.0, x - self.x1)
        dy = max(self.y0 - y, 0.0, y - self.y1)
        return -float(np.hypot(dx, dy))


def playing_area(side: Side, doubles: bool = False) -> Rect:
    half_w = HALF_DW if doubles else HALF_SW
    if side is Side.NEAR:
        return Rect(-half_w, half_w, -HALF_L, 0.0)
    return Rect(-half_w, half_w, 0.0, HALF_L)


def service_box(server_side: Side, box: ServeBox) -> Rect:
    """The box a serve must land in.

    The serve goes diagonally: a near server serving to the deuce court aims at
    the far half's box on the receiver's right, which is -x from the camera.
    """
    target_left = (server_side is Side.NEAR) == (box is ServeBox.DEUCE)
    x0, x1 = (-HALF_SW, 0.0) if target_left else (0.0, HALF_SW)
    if server_side is Side.NEAR:
        return Rect(x0, x1, 0.0, SERVICE_LINE)
    return Rect(x0, x1, -SERVICE_LINE, 0.0)


def side_of(y: float) -> Side:
    return Side.NEAR if y < 0 else Side.FAR


class CourtHomography:
    """Maps between image pixels and court meters for the ground plane."""

    def __init__(self, image_corners: np.ndarray):
        corners = np.asarray(image_corners, dtype=np.float64).reshape(4, 2)
        self.image_corners = corners
        self.court_to_image = cv2.getPerspectiveTransform(
            DOUBLES_CORNERS.astype(np.float32), corners.astype(np.float32)
        )
        self.image_to_court = np.linalg.inv(self.court_to_image)

    @staticmethod
    def _apply(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        homo = np.hstack([pts, np.ones((len(pts), 1))]) @ h.T
        return homo[:, :2] / homo[:, 2:3]

    def to_court(self, pts: np.ndarray) -> np.ndarray:
        return self._apply(self.image_to_court, pts)

    def to_image(self, pts: np.ndarray) -> np.ndarray:
        return self._apply(self.court_to_image, pts)

    def draw(self, frame: np.ndarray, color=(0, 255, 255), thickness: int = 2) -> np.ndarray:
        out = frame.copy()
        for a, b in COURT_LINES:
            p = self.to_image(np.array([a, b]))
            cv2.line(out, tuple(np.round(p[0]).astype(int)), tuple(np.round(p[1]).astype(int)), color, thickness)
        return out


class CourtCamera:
    """Full pinhole camera recovered from the court homography.

    Assumes square pixels and the principal point at the image center, which
    holds well for phone cameras. That leaves the focal length as the only
    intrinsic unknown, and the court's right angles pin it down.
    """

    def __init__(self, homography: CourtHomography, image_size: tuple[int, int]):
        width, height = image_size
        self.cx, self.cy = width / 2, height / 2
        t = np.array([[1, 0, -self.cx], [0, 1, -self.cy], [0, 0, 1.0]])
        h = t @ homography.court_to_image
        h1, h2, h3 = h[:, 0], h[:, 1], h[:, 2]
        # Two constraints on f^2, each linear: D * f^2 + N = 0, from r1 . r2 = 0 and
        # |r1| = |r2|. Seen square-on from behind a baseline the first one's D
        # vanishes and its estimate is noise, so solve both in least squares,
        # which weighs each by how well it is conditioned.
        d = np.array([h1[2] * h2[2], h1[2] ** 2 - h2[2] ** 2])
        n = np.array([h1[0] * h2[0] + h1[1] * h2[1], h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2])
        f2 = -float(d @ n) / float(d @ d) if d @ d > 1e-24 else -1.0
        # Fall back to a typical phone field of view when the geometry is degenerate.
        self.f = float(np.sqrt(f2)) if f2 > 0 else width / (2 * np.tan(np.deg2rad(35)))
        k_inv = np.diag([1 / self.f, 1 / self.f, 1.0])
        r1, r2, tr = k_inv @ h1, k_inv @ h2, k_inv @ h3
        lam = 1 / np.linalg.norm(r1)
        r1, r2, tr = r1 * lam, r2 * lam, tr * lam
        r2 = r2 / np.linalg.norm(r2)
        r3 = np.cross(r1, r2)
        rot = np.stack([r1, r2, r3], axis=1)
        u, _, vt = np.linalg.svd(rot)
        rot = u @ vt
        center = -rot.T @ tr
        if center[2] < 0:  # camera must be above the ground
            rot[:, :2] *= -1
            tr = -tr
            rot[:, 2] = np.cross(rot[:, 0], rot[:, 1])
            center = -rot.T @ tr
        self.R, self.t, self.center = rot, tr, center
        k = np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1.0]])
        self.P = k @ np.hstack([rot, tr[:, None]])

    def project(self, pts: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
        homo = np.hstack([pts, np.ones((len(pts), 1))]) @ self.P.T
        return homo[:, :2] / homo[:, 2:3]
