"""Rally highlights from a match video: find the points, follow the ball, draw it.

1. Court shots and a quick first ball track (`broadcast.find_court_shots`,
   `broadcast.track_shot` with the motion detector) give the stretches where
   the ball is in play.
2. With TrackNet weights, each of those stretches is tracked again with
   TrackNet. Only those stretches: it is slow without a GPU.
3. Contacts are read from the image track, and each point starts at its serve
   (`events.detect_events_2d`). A point ends at its first long lull.
4. The points are joined into one H.264 video with the ball's trail, every
   bounce's call, a small court map of where the ball landed, and each shot's
   estimated speed off the racket.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from .broadcast import CourtShot, _analysis_size, find_court_shots, track_shot
from .court import COURT_LINES, HALF_DW, HALF_L, HALF_SW, SERVICE_LINE, CourtHomography
from .events import SERVE_CONTACT_M, Event, detect_events_2d, remove_spikes, split_rallies

LEAD_IN_S = 1.0  # shown before the serve (the toss)
TAIL_S = 1.2  # shown after the last contact
LULL_S = 2.0  # this long without a contact ends the point
PAD_S = 0.7  # extra frames tracked around each rally for the second pass


@dataclass
class Point:
    start: int  # first frame shown, source video
    end: int  # frame after the last one shown
    events: list[Event]  # frames relative to `start`, image points in `start`-frame pixels
    track: np.ndarray  # ball per frame from `start`, in `start`-frame analysis pixels
    to_ref: list[np.ndarray]  # per frame: maps it onto the `start` frame
    serve_seen: bool
    speeds: dict[int, float] = field(default_factory=dict)  # event index of a bounce -> km/h of its shot


def find_points(
    path: str,
    tracknet_weights: str | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[list[Point], dict]:
    """The points in a video, each from its serve to its end. Returns (points, video info)."""

    def stage(lo: float, hi: float, text: str):
        return (lambda p: progress(lo + (hi - lo) * p, text)) if progress else None

    shots, fps, n_frames = find_court_shots(path, progress=stage(0.0, 0.25, "코트 화면 찾는 중"))
    cap = cv2.VideoCapture(path)
    scale, size = _analysis_size(cap)
    info = {
        "fps": fps,
        "frames": n_frames,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "analysis_scale": scale,
        "court_shots": len(shots),
        "tracker": "tracknet" if tracknet_weights else "motion",
    }

    # First pass: where is the ball in play at all?
    spans: list[tuple[CourtShot, int, int]] = []  # shot, first frame, end frame (source)
    total = sum(s.end - s.start for s in shots) or 1
    done = 0
    first_pass: dict[int, tuple[np.ndarray, list[np.ndarray]]] = {}
    for k, shot in enumerate(shots):
        report = stage(0.25 + 0.25 * done / total, 0.25 + 0.25 * (done + shot.end - shot.start) / total, "랠리 찾는 중")
        track, _, _, to_refs = track_shot(cap, shot, size, progress=report)
        done += shot.end - shot.start
        first_pass[k] = (track, to_refs)
        for a, b in split_rallies(remove_spikes(track, 8.0 * size[0] / 1280), fps, min_len_s=1.0):
            spans.append((k, shot.start + a, shot.start + b))

    # Second pass on the rallies only, with the better detector if there is one.
    detector = None
    if tracknet_weights:
        from .tracknet import TrackNetDetector

        detector = TrackNetDetector(tracknet_weights)
    points: list[Point] = []
    pad = int(round(PAD_S * fps))
    total = sum(b - a + 2 * pad for _, a, b in spans) or 1
    done = 0
    for k, a, b in spans:
        shot = shots[k]
        shot_track, shot_maps = first_pass[k]
        a, b = max(shot.start, a - pad), min(shot.start + len(shot_maps), b + pad)
        if b - a < fps:
            continue
        report = stage(0.5 + 0.45 * done / total, 0.5 + 0.45 * (done + b - a) / total, "공 추적 중")
        done += b - a
        # Re-anchor on this stretch's first frame: its court, and each frame's mapping onto it.
        to_a = np.linalg.inv(shot_maps[a - shot.start])
        corners = CourtHomography._apply(to_a, shot.corners)
        if detector is not None:
            track, ref, _, to_refs = track_shot(cap, CourtShot(a, b, corners, shot.score), size, detector, report)
        else:
            ref = CourtHomography(corners)
            seg = shot_track[a - shot.start : b - shot.start]
            track = np.full_like(seg, np.nan)
            seen = ~np.isnan(seg[:, 0])
            if seen.any():
                track[seen] = CourtHomography._apply(to_a, seg[seen])
            to_refs = [to_a @ m for m in shot_maps[a - shot.start : b - shot.start]]
        points += _points_in(track, ref, to_refs, a, fps, size)
    cap.release()
    points.sort(key=lambda p: p.start)
    return points, info


def _points_in(
    track: np.ndarray, ref: CourtHomography, to_refs: list[np.ndarray], first: int, fps: float, size: tuple[int, int]
) -> list[Point]:
    track = remove_spikes(track, tol_px=max(6.0, 8.0 * size[0] / 1280))
    out = []
    for a, b in split_rallies(track, fps):
        events = detect_events_2d(track, a, b, ref, 1.5 * size[0] / 1280, fps)
        serve_seen = bool(events) and events[0].kind == "hit" and events[0].height_m == SERVE_CONTACT_M
        unseen_serve = (
            bool(events)
            and events[0].kind == "bounce"
            and abs(events[0].court_xy[1]) <= SERVICE_LINE + 1.5
            and abs(events[0].court_xy[0]) <= HALF_SW + 1.0
        )
        if not (serve_seen or unseen_serve):
            continue  # no serve: the tail of a point, or balls knocked about between points
        end = len(events)
        for n in range(1, len(events)):
            if (events[n].frame - events[n - 1].frame) / fps > LULL_S:
                end = n
                break
        events = events[:end]
        start = max(0, int(events[0].frame - LEAD_IN_S * fps))
        stop = min(len(track), len(to_refs), int(events[-1].frame + TAIL_S * fps))
        if stop - start < fps:
            continue
        for e in events:
            e.frame -= start
        speeds = {
            n: events[n - 1].speed_kmh
            for n in range(1, len(events))
            if events[n].kind == "bounce" and events[n - 1].kind == "hit" and events[n - 1].speed_kmh
        }
        out.append(
            Point(
                first + start,
                first + stop,
                events,
                track[start:stop],
                to_refs[start:stop],
                serve_seen,
                speeds,
            )
        )
    return out


def is_in(e: Event, serve: bool = False) -> bool:
    """In the singles court, or for a serve in the service boxes (which one is not known
    without the score, so either)."""
    return abs(e.court_xy[0]) <= HALF_SW and abs(e.court_xy[1]) <= (SERVICE_LINE if serve else HALF_L)


def _serve_landing(point: "Point") -> int | None:
    """Index of the event where the serve landed."""
    n = 1 if point.serve_seen else 0
    return n if n < len(point.events) and point.events[n].kind == "bounce" else None


def render(path: str, points: list[Point], info: dict, out: str, progress: Callable[[float], None] | None = None):
    """Joins the points into one H.264 video with the ball, calls, court map and speeds drawn on."""
    import imageio_ffmpeg

    fps, width, height = info["fps"], info["width"], info["height"]
    scale = info["analysis_scale"]
    writer = imageio_ffmpeg.write_frames(
        out,
        (width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_out="yuv420p",
        output_params=["-crf", "23", "-preset", "veryfast", "-movflags", "+faststart"],
        macro_block_size=2,
        ffmpeg_log_level="error",
    )
    writer.send(None)
    cap = cv2.VideoCapture(path)
    total = sum(p.end - p.start for p in points) or 1
    done = 0
    court_map = _CourtMap(width, height)
    for k, point in enumerate(points):
        cap.set(cv2.CAP_PROP_POS_FRAMES, point.start)
        for i in range(point.end - point.start):
            ok, img = cap.read()
            if not ok:
                break
            _draw_point_frame(img, point, i, scale, court_map, f"Point {k + 1}/{len(points)}")
            writer.send(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).tobytes())
            done += 1
            if progress and done % 30 == 0:
                progress(done / total)
    cap.release()
    writer.close()


def _draw_point_frame(img: np.ndarray, point: Point, i: int, scale: float, court_map: "_CourtMap", title: str):
    # Stabilized (first-frame) analysis pixels -> this frame's source pixels.
    back = np.linalg.inv(point.to_ref[min(i, len(point.to_ref) - 1)])

    def here(xy) -> tuple[int, int]:
        p = CourtHomography._apply(back, np.array([xy], dtype=np.float64))[0] / scale
        return int(round(p[0])), int(round(p[1]))

    trail = [here(p) for p in point.track[max(0, i - 10) : i + 1] if not np.isnan(p[0])]
    for j, p in enumerate(trail):
        cv2.circle(img, p, 2 + j // 3, (0, 215, 255), -1, cv2.LINE_AA)
    marks, speed, speed_label, caption = [], None, "", ""
    serve_n = _serve_landing(point)
    for n, e in enumerate(point.events):
        if e.frame > i:
            break
        if e.kind != "bounce":
            continue
        inside = is_in(e, n == serve_n)
        marks.append((e.court_xy, inside))
        if n in point.speeds:
            speed = point.speeds[n]
            speed_label = "serve" if n == serve_n and point.serve_seen else "last shot"
        if i - e.frame < 30:
            cv2.circle(img, here(e.image_xy), 16, (0, 200, 0) if inside else (0, 0, 255), 3, cv2.LINE_AA)
            caption = f"BOUNCE {'IN' if inside else 'OUT'}"
            if n in point.speeds:
                caption += f"   {speed_label} ~{point.speeds[n]:.0f} km/h"
    court_map.draw(img, marks, speed, speed_label or "last shot")
    h, w = img.shape[:2]
    bar = max(28, h // 24)
    cv2.rectangle(img, (0, h - bar), (w, h), (0, 0, 0), -1)
    cv2.putText(img, f"{title}   {caption}", (10, h - bar // 3), 0, bar / 50, (255, 255, 255), 1, cv2.LINE_AA)


class _CourtMap:
    """Top-down court in the top-right corner with this point's landings."""

    def __init__(self, width: int, height: int):
        u = height / 720
        self.w, self.h, self.m = int(100 * u), int(180 * u), int(12 * u)
        self.x0, self.y0 = width - self.w - 2 * self.m - int(12 * u), int(120 * u)
        self.u = u

    def at(self, x: float, y: float) -> tuple[int, int]:
        return (
            int(self.x0 + self.m + (x + HALF_DW) / (2 * HALF_DW) * self.w),
            int(self.y0 + self.m + (HALF_L - y) / (2 * HALF_L) * self.h),
        )

    def draw(self, img: np.ndarray, marks, speed: float | None, label: str) -> None:
        u = self.u
        x1, y1 = self.x0 + self.w + 2 * self.m, self.y0 + self.h + 2 * self.m + int(40 * u)
        overlay = img.copy()
        cv2.rectangle(overlay, (self.x0, self.y0), (x1, y1), (30, 30, 30), -1)
        img[:] = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
        cv2.rectangle(img, self.at(-HALF_DW, HALF_L), self.at(HALF_DW, -HALF_L), (150, 110, 60), -1)
        for a, b in COURT_LINES:
            cv2.line(img, self.at(*a), self.at(*b), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(img, self.at(-HALF_DW - 0.9, 0), self.at(HALF_DW + 0.9, 0), (200, 200, 200), 2)
        for j, ((x, y), inside) in enumerate(marks):
            p = self.at(np.clip(x, -HALF_DW - 1.5, HALF_DW + 1.5), np.clip(y, -HALF_L - 3, HALF_L + 3))
            last = j == len(marks) - 1
            cv2.circle(img, p, int((5 if last else 3) * u) + 1, (0, 220, 0) if inside else (0, 0, 255), -1, cv2.LINE_AA)
            if last:
                cv2.circle(img, p, int(7 * u) + 1, (255, 255, 255), 1, cv2.LINE_AA)
        base = self.y0 + self.h + 2 * self.m
        cv2.putText(img, label, (self.x0 + 6, base + int(14 * u)), 0, 0.4 * u, (200, 200, 200), 1, cv2.LINE_AA)
        text = f"{speed:.0f} km/h" if speed else "-"
        cv2.putText(img, text, (self.x0 + 6, base + int(33 * u)), 0, 0.55 * u, (255, 255, 255), 1, cv2.LINE_AA)


def summarize(points: list[Point], info: dict) -> dict:
    """The report the app shows next to the highlight video."""
    fps = info["fps"]
    out, clip_t = [], 0.0
    for k, p in enumerate(points):
        bounces = []
        serve_n = _serve_landing(p)
        for n, e in enumerate(p.events):
            if e.kind != "bounce":
                continue
            bounces.append(
                {
                    "clip_t": round(clip_t + e.frame / fps, 2),
                    "source_t": round((p.start + e.frame) / fps, 2),
                    "court_xy": [round(v, 2) for v in e.court_xy],
                    "in": is_in(e, n == serve_n),
                    "serve": n == serve_n,
                    "speed_kmh": p.speeds.get(n),
                }
            )
        serve_speed = p.speeds.get(1) if p.serve_seen else None
        out.append(
            {
                "index": k,
                "clip_start_t": round(clip_t, 2),
                "clip_end_t": round(clip_t + (p.end - p.start) / fps, 2),
                "source_start_t": round(p.start / fps, 2),
                "source_end_t": round(p.end / fps, 2),
                "serve_seen": p.serve_seen,
                "serve_speed_kmh": serve_speed,
                "shots": sum(1 for e in p.events if e.kind == "hit") + (0 if p.serve_seen else 1),
                "bounces": bounces,
            }
        )
        clip_t += (p.end - p.start) / fps
    speeds = [s for p in points for s in p.speeds.values()]
    serves = [q["serve_speed_kmh"] for q in out if q["serve_speed_kmh"]]
    return {
        "video": info,
        "points": out,
        "stats": {
            "points": len(out),
            "highlight_s": round(clip_t, 1),
            "bounces": sum(len(q["bounces"]) for q in out),
            "bounces_in": sum(b["in"] for q in out for b in q["bounces"]),
            "serve_speed_max_kmh": max(serves, default=None),
            "serve_speed_avg_kmh": round(float(np.mean(serves)), 1) if serves else None,
            "shot_speed_avg_kmh": round(float(np.mean(speeds)), 1) if speeds else None,
        },
    }


def make_highlights(
    path: str,
    out_video: str,
    tracknet_weights: str | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> dict:
    """Finds the points in `path`, writes their highlight video to `out_video`, returns the report."""
    points, info = find_points(path, tracknet_weights, progress)
    render(path, points, info, out_video, (lambda p: progress(0.95 + 0.05 * p, "영상 만드는 중")) if progress else None)
    return summarize(points, info)


def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Cut a match video down to its points, with the ball tracked")
    ap.add_argument("video")
    ap.add_argument("--out", default="highlights.mp4")
    ap.add_argument("--report", help="also write the report JSON here")
    ap.add_argument("--tracknet", metavar="WEIGHTS", help="find the ball with TrackNet (needs torch)")
    args = ap.parse_args()
    report = make_highlights(args.video, args.out, args.tracknet, lambda p, s: print(f"{p:5.1%} {s}", flush=True))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        with open(args.report, "w") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
