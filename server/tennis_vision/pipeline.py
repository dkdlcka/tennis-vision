"""End-to-end analysis of one video: court, ball, bounces, calls, score, stats."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .ball import BallTracker, MotionColorDetector
from .court import SERVICE_LINE, CourtHomography, side_of
from .court_detect import detect_court
from .events import find_rallies
from .scoring import Match, judge_point
from .shots import miss_type, serve_zone, shot_direction

MAX_WIDTH = 1280


@dataclass
class Options:
    corners: list[list[float]] | None = None  # 4 doubles corners in source pixels: NL, NR, FR, FL
    doubles: bool = False
    best_of: int = 3
    no_ad: bool = False
    first_server: str = "A"
    a_starts_near: bool = True
    player_names: list[str] = field(default_factory=lambda: ["A", "B"])


def _read_frames(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return cap, fps, total


def analyze_video(path: str, options: Options | None = None, progress: Callable[[float], None] | None = None) -> dict:
    options = options or Options()
    cap, fps, total = _read_frames(path)
    ok, first = cap.read()
    if not ok:
        raise ValueError("video has no frames")
    src_h, src_w = first.shape[:2]
    scale = min(1.0, MAX_WIDTH / src_w)
    size = (int(round(src_w * scale)), int(round(src_h * scale)))

    def prep(frame: np.ndarray) -> np.ndarray:
        return cv2.resize(frame, size, interpolation=cv2.INTER_AREA) if scale != 1.0 else frame

    first_small = prep(first)
    court_source = "manual"
    court_score = None
    if options.corners:
        homography = CourtHomography(np.array(options.corners, dtype=np.float64) * scale)
    else:
        found = detect_court(first_small)
        if found is None:
            raise CourtNotFound("could not find the court automatically; pass corners")
        homography, court_score = found
        court_source = "auto"

    detector = MotionColorDetector((size[1], size[0]))
    tracker = BallTracker(gate_px=60 * size[0] / 1280)
    frame_idx = 0
    frame = first_small
    while True:
        tracker.update(frame_idx, detector.detect(frame))
        frame_idx += 1
        if progress and total and frame_idx % 30 == 0:
            progress(min(0.95, frame_idx / total))
        ok, raw = cap.read()
        if not ok:
            break
        frame = prep(raw)
    cap.release()
    n_frames = frame_idx
    track = tracker.result(n_frames)

    rallies = find_rallies(track, homography, fps, size)
    report = build_report(rallies, track, homography, fps, n_frames, options)
    report["video"] = {"fps": fps, "frames": n_frames, "width": src_w, "height": src_h, "analysis_scale": scale}
    report["court"] = {
        "source": court_source,
        "score": court_score,
        "corners_px": (homography.image_corners / scale).round(1).tolist(),
    }
    # Ball track in source pixels, for drawing on the original video in the app.
    report["ball_track"] = [
        [i, round(float(x / scale), 1), round(float(y / scale), 1)] for i, (x, y) in enumerate(track) if not np.isnan(x)
    ]
    if progress:
        progress(1.0)
    return report


class CourtNotFound(ValueError):
    pass


def build_report(rallies, track, homography, fps, n_frames, options: Options) -> dict:
    match = Match(
        best_of=options.best_of,
        no_ad=options.no_ad,
        first_server=options.first_server,
        a_starts_near=options.a_starts_near,
    )
    names = {"A": options.player_names[0], "B": options.player_names[1]}
    points = []
    bounces_out = []
    stats = {p: _empty_stats() for p in ("A", "B")}

    for r_idx, rally in enumerate(rallies):
        if match.winner:
            break
        server = match.current_server
        server_side = match.side_of_player(server)
        box = match.serve_box
        was_second = match.second_serve
        bounce_events = rally.bounces
        result = judge_point([b.court_xy for b in bounce_events], server_side, box, options.doubles)
        if result.winner is None and not result.fault:
            continue

        # Who hit each ball: serve by the server, then the player opposite each bounce.
        bounce_ids = []
        prev_frame = -1.0
        for ev, call in zip(bounce_events, result.calls):
            hitter = server if call.kind == "serve" else match.player_on(call.side.other)
            # The last racket contact before this bounce, if the ball was seen leaving the racket.
            contact = [e for e in rally.events if e.kind == "hit" and prev_frame < e.frame < ev.frame]
            prev_frame = ev.frame
            detail = {}
            if call.kind == "serve":
                if call.inside:
                    detail["zone"] = serve_zone(call.x)
                    stats[hitter]["serve_zones"][detail["zone"]] += 1
            else:
                detail["direction"] = shot_direction(contact[-1].court_xy[0] if contact else None, call.x)
                if detail["direction"]:
                    stats[hitter]["directions"][detail["direction"]] += 1
            if not call.inside:
                own_half = call.side is (server_side if call.kind == "serve" else match.side_of_player(hitter))
                detail["miss"] = miss_type(call.x, call.y, call.kind == "serve", options.doubles, own_half)
            bounce_ids.append(len(bounces_out))
            bounces_out.append(
                {
                    "id": len(bounces_out),
                    "frame": round(ev.frame, 2),
                    "t": round(ev.frame / fps, 3),
                    "image_xy": [round(v, 1) for v in ev.image_xy],
                    "court_xy": [round(call.x, 3), round(call.y, 3)],
                    "side": call.side.value,
                    "in": call.inside,
                    "margin_cm": round(call.margin_m * 100, 1),
                    "kind": call.kind,
                    "hitter": hitter,
                    "rally": r_idx,
                    **detail,
                }
            )
            s = stats[hitter]
            if call.kind == "rally":
                s["rally_shots"] += 1
                if call.inside:
                    s["rally_in"] += 1
                    if abs(call.y) > SERVICE_LINE:
                        s["deep_shots"] += 1
                s["landing"].append([round(call.x, 2), round(call.y, 2)])

        if rally.arcs:
            serve_speed = round(float(np.linalg.norm(rally.arcs[0].v0)) * 3.6, 1)
        else:  # contacts read from the image track: the serve's own estimate, if any
            first = rally.events[0] if rally.events else None
            serve_speed = first.speed_kmh if first and first.kind == "hit" else None
        if serve_speed is not None:
            stats[server]["serve_speeds"].append(serve_speed)
        hits = []
        for ev in rally.events:
            if ev.kind != "hit" or ev.speed_kmh is None:
                continue
            hitter = match.player_on(side_of(ev.court_xy[1]))
            stats[hitter]["shot_speeds"].append(ev.speed_kmh)
            hits.append(
                {
                    "t": round(ev.frame / fps, 3),
                    "player": hitter,
                    "speed_kmh": ev.speed_kmh,
                    "height_m": round(ev.height_m, 2),
                }
            )
        serve_stat = stats[server]
        serve_stat["serves"] += 1
        if was_second:
            serve_stat["second_serves"] += 1
        else:
            serve_stat["first_serves"] += 1
        if result.fault:
            double = match.record_fault()
            if double:
                serve_stat["double_faults"] += 1
            outcome = {"winner": _other(server) if double else None, "reason": "double_fault" if double else "fault"}
        else:
            if was_second:
                serve_stat["second_serves_in"] += 1
            else:
                serve_stat["first_serves_in"] += 1
            winner = match.player_on(result.winner)
            if result.reason == "ace":
                serve_stat["aces"] += 1
            if result.reason == "out":
                stats[_other(winner)]["errors_out"] += 1
                last = bounces_out[-1]
                stats[_other(winner)][f"errors_{last['miss']}"] += 1
            if result.reason == "double_bounce":
                stats[_other(winner)]["errors_net_or_missed"] += 1
            if result.reason == "winner":
                stats[winner]["winners"] += 1
            match.award(winner)
            outcome = {"winner": winner, "reason": result.reason}

        points.append(
            {
                "index": len(points),
                "start_t": round(rally.start_frame / fps, 3),
                "end_t": round(rally.end_frame / fps, 3),
                "server": server,
                "server_side": server_side.value,
                "serve_box": box.value,
                "second_serve": was_second,
                **outcome,
                "shots": len(bounce_events),
                "bounce_ids": bounce_ids,
                "serve_speed_kmh": serve_speed,
                "hits": hits,
                "net_clearance_m": rally.net_clearances(),
                "score_after": match.display(),
            }
        )

    for p, s in stats.items():
        s["first_serve_pct"] = _pct(s["first_serves_in"], s["first_serves"])
        s["rally_in_pct"] = _pct(s["rally_in"], s["rally_shots"])
        s["deep_pct"] = _pct(s["deep_shots"], s["rally_in"])
        s["points_won"] = sum(1 for pt in points if pt["winner"] == p)
        s["serve_speed_avg_kmh"] = _avg(s["serve_speeds"])
        s["serve_speed_max_kmh"] = max(s["serve_speeds"], default=None)
        s["shot_speed_avg_kmh"] = _avg(s["shot_speeds"])

    rally_lengths = [pt["shots"] for pt in points if pt["winner"]]
    return {
        "players": names,
        "points": points,
        "bounces": bounces_out,
        "score": match.display(),
        "stats": {
            "players": stats,
            "rallies": len(rally_lengths),
            "avg_rally_shots": round(float(np.mean(rally_lengths)), 2) if rally_lengths else 0,
            "longest_rally_shots": max(rally_lengths, default=0),
            "close_calls": [b["id"] for b in bounces_out if abs(b["margin_cm"]) < 5],
        },
    }


def _avg(v: list[float]) -> float | None:
    return round(float(np.mean(v)), 1) if v else None


def _empty_stats() -> dict:
    return {
        "serves": 0,
        "first_serves": 0,
        "first_serves_in": 0,
        "second_serves": 0,
        "second_serves_in": 0,
        "aces": 0,
        "double_faults": 0,
        "rally_shots": 0,
        "rally_in": 0,
        "deep_shots": 0,
        "winners": 0,
        "errors_out": 0,
        "errors_net_or_missed": 0,
        "landing": [],
        "serve_zones": {"wide": 0, "body": 0, "T": 0},
        "directions": {"cross": 0, "line": 0, "center": 0},
        "errors_long": 0,
        "errors_wide": 0,
        "errors_net": 0,
        "serve_speeds": [],
        "shot_speeds": [],
    }


def _pct(a: int, b: int) -> float | None:
    return round(100 * a / b, 1) if b else None


def _other(p: str) -> str:
    return "B" if p == "A" else "A"


def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Analyze a tennis video")
    ap.add_argument("video")
    ap.add_argument("--corners", help="NLx,NLy,NRx,NRy,FRx,FRy,FLx,FLy doubles corners in pixels")
    ap.add_argument("--doubles", action="store_true")
    ap.add_argument("--out", default="-")
    args = ap.parse_args()
    corners = None
    if args.corners:
        v = [float(x) for x in args.corners.split(",")]
        corners = [v[i : i + 2] for i in range(0, 8, 2)]
    report = analyze_video(args.video, Options(corners=corners, doubles=args.doubles))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out == "-":
        print(text)
    else:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
