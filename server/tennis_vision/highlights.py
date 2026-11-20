"""Automatic editing: pick highlight points and cut videos that skip the dead time.

From a finished report this module
- tags each point worth watching again (ace, winner, long rally, close line call,
  fastest serve) and ranks them,
- plans the segments of a condensed video that keeps only the points,
- cuts those segments out of the original video with ffmpeg.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

PRE_ROLL_S = 1.5  # keep the serve toss before the ball is first seen
POST_ROLL_S = 1.5  # and the reaction after the last bounce
MERGE_GAP_S = 1.0  # segments closer than this are joined into one
LONG_RALLY_BOUNCES = 6  # a point with at least this many bounces counts as a long rally
CLOSE_CALL_CM = 5.0
HIGHLIGHT_REEL_MAX = 8

TAG_WEIGHT = {
    "ace": 3.0,
    "winner": 3.0,
    "long_rally": 2.0,
    "close_call": 2.0,
    "fastest_serve": 1.5,
    "game_point": 1.0,
}

TAG_LABEL = {
    "ace": "에이스",
    "winner": "위너",
    "long_rally": "긴 랠리",
    "close_call": "접전 판정",
    "fastest_serve": "최고 서브 속도",
    "game_point": "게임 결정",
}


def clip_window(point: dict, duration: float | None) -> tuple[float, float]:
    start = max(0.0, point["start_t"] - PRE_ROLL_S)
    end = point["end_t"] + POST_ROLL_S
    if duration is not None:
        end = min(end, duration)
    return round(start, 3), round(end, 3)


def pick_highlights(report: dict, duration: float | None = None) -> list[dict]:
    """Points worth watching again, best first."""
    points = report["points"]
    bounces = {b["id"]: b for b in report["bounces"]}
    speeds = [p["serve_speed_kmh"] for p in points if p["serve_speed_kmh"] and p["reason"] != "fault"]
    top_speed = max(speeds, default=None)
    rally_lengths = [p["shots"] for p in points if p["winner"]]
    longest = max(rally_lengths, default=0)

    out = []
    prev_games = None
    for p in points:
        tags = []
        if p["reason"] == "ace":
            tags.append("ace")
        if p["reason"] == "winner":
            tags.append("winner")
        if p["winner"] and p["shots"] >= LONG_RALLY_BOUNCES:
            tags.append("long_rally")
        close = [bounces[i] for i in p["bounce_ids"] if abs(bounces[i]["margin_cm"]) < CLOSE_CALL_CM]
        if close:
            tags.append("close_call")
        if top_speed is not None and p["serve_speed_kmh"] == top_speed and p["reason"] != "fault":
            tags.append("fastest_serve")
        games = _games_played(p["score_after"])
        if prev_games is not None and games > prev_games:
            tags.append("game_point")
        if p["winner"]:
            prev_games = games
        if not tags:
            continue
        score = sum(TAG_WEIGHT[t] for t in tags)
        if longest and p["winner"]:
            score += p["shots"] / longest  # longer rallies first among equals
        start, end = clip_window(p, duration)
        out.append(
            {
                "point": p["index"],
                "start_t": start,
                "end_t": end,
                "tags": tags,
                "labels": [TAG_LABEL[t] for t in tags],
                "score": round(score, 2),
                "close_call_cm": min((abs(b["margin_cm"]) for b in close), default=None),
            }
        )
    out.sort(key=lambda h: (-h["score"], h["point"]))
    return out


def condensed_segments(points: list[dict], duration: float | None = None) -> list[tuple[float, float]]:
    """Time ranges of the original video that hold play, merged where they nearly touch."""
    segs: list[list[float]] = []
    for p in sorted(points, key=lambda p: p["start_t"]):
        start, end = clip_window(p, duration)
        if segs and start - segs[-1][1] < MERGE_GAP_S:
            segs[-1][1] = max(segs[-1][1], end)
        else:
            segs.append([start, end])
    return [(round(a, 3), round(b, 3)) for a, b in segs]


def editing_plan(report: dict, duration: float | None) -> dict:
    """What goes into the report: highlights and the condensed video's timeline."""
    highlights = pick_highlights(report, duration)
    segs = condensed_segments(report["points"], duration)
    kept = sum(b - a for a, b in segs)
    reel = sorted(highlights[:HIGHLIGHT_REEL_MAX], key=lambda h: h["start_t"])
    return {
        "highlights": highlights,
        "condensed": {
            "segments": [list(s) for s in segs],
            "duration_s": round(kept, 2),
            "original_s": round(duration, 2) if duration is not None else None,
        },
        "reel_points": [h["point"] for h in reel],
    }


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def cut_video(src: str | Path, segments: list[tuple[float, float]], out: str | Path) -> Path:
    """Writes the given time ranges of `src`, joined in order, to `out` (H.264 mp4)."""
    if not segments:
        raise ValueError("no segments to cut")
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg is not installed")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        parts = []
        for i, (a, b) in enumerate(segments):
            part = Path(tmp) / f"part{i:04d}.mp4"
            # Re-encode each part so every cut lands on the exact frame and parts share one format.
            _run_ffmpeg(
                ["-ss", f"{a:.3f}", "-i", str(src), "-t", f"{b - a:.3f}", "-map", "0:v:0", "-map", "0:a:0?"]
                + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p"]
                + ["-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2", "-avoid_negative_ts", "make_zero"]
                + [str(part)]
            )
            parts.append(part)
        if len(parts) == 1:
            shutil.move(parts[0], out)
            return out
        listing = Path(tmp) / "parts.txt"
        listing.write_text("".join(f"file '{p.name}'\n" for p in parts))
        _run_ffmpeg(
            ["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(out)]
        )
    return out


def _run_ffmpeg(args: list[str]) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()[-500:]}")


def _games_played(score: dict) -> int:
    return sum(a + b for a, b in score["sets"])


def reel_segments(report: dict) -> list[tuple[float, float]]:
    """Time ranges of the highlight reel, in match order."""
    by_point = {h["point"]: h for h in report["highlights"]}
    return [(by_point[i]["start_t"], by_point[i]["end_t"]) for i in report["reel_points"]]


def point_segment(report: dict, index: int) -> tuple[float, float]:
    duration = report["video"]["frames"] / report["video"]["fps"]
    return clip_window(report["points"][index], duration)
