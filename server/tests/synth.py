"""Renders synthetic tennis videos with a known ground truth.

A pinhole camera sits behind the near baseline. Shots are scripted by where
they land, and the ball follows simple projectile physics with a lossy bounce.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from tennis_vision.court import COURT_LINES, HALF_DW, HALF_L

G = 9.81


@dataclass
class Camera:
    width: int = 1280
    height: int = 720
    position: tuple[float, float, float] = (0.0, -HALF_L - 9.0, 7.0)
    target: tuple[float, float, float] = (0.0, -3.0, 0.0)
    hfov_deg: float = 62.0

    def __post_init__(self) -> None:
        c = np.array(self.position)
        fwd = np.array(self.target) - c
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        self.r = np.stack([right, -up, fwd])  # camera x right, y down, z forward
        self.c = c
        self.f = (self.width / 2) / np.tan(np.deg2rad(self.hfov_deg / 2))

    def project(self, pts: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
        if pts.shape[1] == 2:
            pts = np.hstack([pts, np.zeros((len(pts), 1))])
        cam = (pts - self.c) @ self.r.T
        return np.stack(
            [self.f * cam[:, 0] / cam[:, 2] + self.width / 2, self.f * cam[:, 1] / cam[:, 2] + self.height / 2], axis=1
        )

    def depth(self, p: np.ndarray) -> float:
        return float(((np.asarray(p) - self.c) @ self.r.T)[2])


@dataclass
class Shot:
    land: tuple[float, float]
    flight: float  # seconds from contact to bounce


@dataclass
class ScriptedPoint:
    start: tuple[float, float, float]  # contact point of the serve
    shots: list[Shot]
    tail: float = 0.7  # seconds the ball stays in view after the last bounce
    double_bounce: bool = False  # after the last bounce let it bounce again instead of ending


@dataclass
class Truth:
    positions: list[np.ndarray | None] = field(default_factory=list)  # per frame, 3D or None
    bounces: list[tuple[int, float, float]] = field(default_factory=list)  # (frame, x, y)


def simulate(points: list[ScriptedPoint], fps: float, gap_s: float = 2.0, lead_s: float = 1.0) -> Truth:
    """Samples the scripted points at `fps`. Bounce frames are fractional-exact, rounded."""
    # Continuous-time segments: (t_start, duration, p0 xyz, v xy, vz)
    segments = []
    bounce_times = []
    t = lead_s
    for sp in points:
        p = np.array(sp.start, dtype=np.float64)
        for k, shot in enumerate(sp.shots):
            land = np.array([*shot.land, 0.0])
            v = (land[:2] - p[:2]) / shot.flight
            vz = (0.5 * G * shot.flight**2 - p[2]) / shot.flight
            segments.append((t, shot.flight, p.copy(), v, vz))
            t += shot.flight
            bounce_times.append((t, land[0], land[1]))
            v_out = v * 0.75
            vz_out = -(vz - G * shot.flight) * 0.7
            last = k == len(sp.shots) - 1
            if last and sp.double_bounce:
                d = 2 * vz_out / G
                segments.append((t, d, land.copy(), v_out, vz_out))
                t += d
                land2 = np.array([*(land[:2] + v_out * d), 0.0])
                bounce_times.append((t, land2[0], land2[1]))
                segments.append((t, sp.tail, land2, v_out * 0.7, vz_out * 0.5))
                t += sp.tail
            elif last:
                segments.append((t, sp.tail, land.copy(), v_out, vz_out))
                t += sp.tail
            else:
                # Rise and fall to a comfortable hitting height of about 1 m.
                disc = vz_out**2 - 2 * G * 1.0
                d = (vz_out + np.sqrt(disc)) / G if disc > 0 else vz_out / G
                # Players take the ball before it runs far behind the baseline.
                room = (HALF_L + 1.0 - abs(land[1])) / max(abs(v_out[1]), 1e-6)
                d = min(d, max(room, 0.25))
                segments.append((t, d, land.copy(), v_out, vz_out))
                t += d
                p = np.array([*(land[:2] + v_out * d), vz_out * d - 0.5 * G * d * d])
        t += gap_s

    n_frames = int(np.ceil(t * fps))
    truth = Truth(positions=[None] * n_frames)
    for t0, d, p0, v, vz in segments:
        for f in range(int(np.ceil(t0 * fps)), int(np.ceil((t0 + d) * fps))):
            s = f / fps - t0
            z = max(0.0, p0[2] + vz * s - 0.5 * G * s * s)
            truth.positions[f] = np.array([p0[0] + v[0] * s, p0[1] + v[1] * s, z])
    truth.bounces = [(int(round(bt * fps)), bx, by) for bt, bx, by in bounce_times]
    return truth


def render(truth: Truth, cam: Camera, seed: int = 0, players: bool = True):
    rng = np.random.default_rng(seed)
    base = np.full((cam.height, cam.width, 3), (70, 120, 60), np.uint8)  # surround (BGR)
    court_poly = cam.project(
        [(-HALF_DW - 3, -HALF_L - 5), (HALF_DW + 3, -HALF_L - 5), (HALF_DW + 3, HALF_L + 5), (-HALF_DW - 3, HALF_L + 5)]
    )
    cv2.fillPoly(base, [court_poly.astype(np.int32)], (140, 90, 50))
    for a, b in COURT_LINES:
        pa, pb = cam.project([a, b])
        # Subpixel endpoints (4 fractional bits) so the painted lines sit exactly on the model.
        pa, pb = (tuple((p * 16).round().astype(int)) for p in (pa, pb))
        cv2.line(base, pa, pb, (245, 245, 245), 3, cv2.LINE_AA, shift=4)
    net_l, net_r = cam.project([(-HALF_DW - 0.9, 0, 1.0), (HALF_DW + 0.9, 0, 1.0)])
    base_l, base_r = cam.project([(-HALF_DW - 0.9, 0, 0), (HALF_DW + 0.9, 0, 0)])
    net = np.array([net_l, net_r, base_r, base_l]).astype(np.int32)
    overlay = base.copy()
    cv2.fillPoly(overlay, [net], (40, 40, 40))
    base = cv2.addWeighted(overlay, 0.35, base, 0.65, 0)
    cv2.line(base, tuple(net_l.astype(int)), tuple(net_r.astype(int)), (250, 250, 250), 2)

    for i, pos in enumerate(truth.positions):
        frame = base.copy()
        if players:
            for py, phase in ((-HALF_L - 0.5, 0.0), (HALF_L + 0.5, 1.3)):
                px = 2.5 * np.sin(i / 25 + phase)
                feet = cam.project([(px, py, 0)])[0]
                head = cam.project([(px, py, 1.8)])[0]
                w = max(4, int((feet[1] - head[1]) * 0.3))
                cv2.rectangle(
                    frame, (int(feet[0] - w), int(head[1])), (int(feet[0] + w), int(feet[1])), (60, 40, 170), -1
                )
        if pos is not None:
            shadow = cam.project([(pos[0], pos[1], 0)])[0]
            cv2.circle(frame, tuple(shadow.round().astype(int)), 3, (100, 70, 40), -1, cv2.LINE_AA)
            c = cam.project([pos])[0]
            r = max(3.0, cam.f * 0.033 / cam.depth(pos))
            cv2.circle(frame, tuple(c.round().astype(int)), int(round(r)), (40, 230, 220), -1, cv2.LINE_AA)
        noise = rng.normal(0, 2.0, frame.shape)
        yield np.clip(frame + noise, 0, 255).astype(np.uint8)


def write_video(path: str, truth: Truth, cam: Camera, fps: float) -> None:
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (cam.width, cam.height))
    for frame in render(truth, cam):
        out.write(frame)
    out.release()


def standard_match() -> list[ScriptedPoint]:
    """Three points with A serving from the near end.

    1. Deuce serve in, rally, B hits long -> A (15-0)
    2. Ad serve long (fault), second serve in, B returns, A lets it bounce twice -> B (15-15)
    3. Deuce serve ace -> A (30-15)
    """
    near_deuce = (1.0, -HALF_L - 0.3, 2.7)
    near_ad = (-1.0, -HALF_L - 0.3, 2.7)
    return [
        ScriptedPoint(
            near_deuce,
            [Shot((-2.0, 5.0), 0.75), Shot((1.5, -8.0), 1.1), Shot((-2.5, 9.0), 1.1), Shot((0.5, -12.6), 1.0)],
        ),
        ScriptedPoint(near_ad, [Shot((2.0, 7.3), 0.75)]),
        ScriptedPoint(near_ad, [Shot((1.8, 4.8), 0.9), Shot((-1.5, -2.0), 0.8)], tail=1.0, double_bounce=True),
        ScriptedPoint(near_deuce, [Shot((-3.4, 5.5), 0.7)], tail=0.8),
    ]
