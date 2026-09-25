"""Undo camera motion by aligning every frame to the first one.

Broadcast cameras pan and zoom a little during a rally, and a phone on a
tripod still shakes. The court is flat and far away, and pure camera rotation
or zoom maps one frame onto another by a homography, so matching features
between each frame and the first gives that homography. Warping frames into
the first frame's pixels keeps the court corners and the background model
valid for the whole video.
"""

from __future__ import annotations

import cv2
import numpy as np

MIN_INLIERS = 40


class FrameAligner:
    def __init__(self, reference: np.ndarray, features: int = 2000):
        self.orb = cv2.ORB_create(features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ref_kp, self.ref_desc = self.orb.detectAndCompute(_gray(reference), None)
        self.last = np.eye(3)

    def align(self, frame: np.ndarray) -> np.ndarray:
        """Homography taking this frame's pixels to the reference frame's pixels.

        Falls back to the previous frame's homography when too few features
        match (a blurred frame, a player filling the view).
        """
        kp, desc = self.orb.detectAndCompute(_gray(frame), None)
        if desc is None or self.ref_desc is None or len(kp) < MIN_INLIERS:
            return self.last
        pairs = self.matcher.knnMatch(desc, self.ref_desc, k=2)
        good = [m for m, *rest in pairs if rest and m.distance < 0.75 * rest[0].distance]
        if len(good) < MIN_INLIERS:
            return self.last
        src = np.float32([kp[m.queryIdx].pt for m in good])
        dst = np.float32([self.ref_kp[m.trainIdx].pt for m in good])
        h, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 1.5)
        if h is None or inliers.sum() < MIN_INLIERS:
            return self.last
        self.last = h / h[2, 2]
        return self.last


def _gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


def apply(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ h.T
    return p[:, :2] / p[:, 2:3]


def is_identity(h: np.ndarray, size: tuple[int, int], tol_px: float = 2.0) -> bool:
    """Whether the frame's corners move less than `tol_px`, about the matching noise on low-texture video."""
    w, hgt = size
    corners = np.array([[0, 0], [w, 0], [w, hgt], [0, hgt]], dtype=np.float64)
    return bool(np.abs(apply(h, corners) - corners).max() < tol_px)
