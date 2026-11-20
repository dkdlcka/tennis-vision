import cv2
import numpy as np

from tennis_vision.highlights import (
    condensed_segments,
    cut_video,
    editing_plan,
    pick_highlights,
    reel_segments,
)


def _point(i, start, end, reason, winner, shots=2, speed=150.0, bounce_ids=(), games=0):
    return {
        "index": i,
        "start_t": start,
        "end_t": end,
        "reason": reason,
        "winner": winner,
        "shots": shots,
        "serve_speed_kmh": speed,
        "bounce_ids": list(bounce_ids),
        "score_after": {"sets": [[games, 0]]},
    }


def _report():
    bounces = [{"id": 0, "margin_cm": 3.0}, {"id": 1, "margin_cm": 80.0}]
    points = [
        _point(0, 2.0, 5.0, "out", "A", shots=3, bounce_ids=[0]),
        _point(1, 10.0, 12.0, "fault", None, speed=190.0),
        _point(2, 14.0, 16.0, "ace", "A", shots=1, speed=170.0),
        _point(3, 30.0, 45.0, "double_bounce", "B", shots=9, bounce_ids=[1]),
        _point(4, 50.0, 53.0, "out", "B", shots=2, games=1),
    ]
    return {"points": points, "bounces": bounces}


def test_tags_and_ranking():
    hl = pick_highlights(_report(), duration=60.0)
    tags = {h["point"]: h["tags"] for h in hl}
    assert tags[0] == ["close_call"]
    assert tags[2] == ["ace", "fastest_serve"]  # the 190 km/h serve was a fault, so it doesn't count
    assert tags[3] == ["long_rally"]
    assert tags[4] == ["game_point"]
    assert 1 not in tags
    assert hl[0]["point"] == 2
    assert hl[0]["labels"] == ["에이스", "최고 서브 속도"]
    assert next(h for h in hl if h["point"] == 0)["close_call_cm"] == 3.0


def test_condensed_merges_close_points_and_clamps():
    segs = condensed_segments(_report()["points"], duration=53.5)
    # Points 1 and 2 are 2 s apart, so with pre/post roll they join into one segment.
    assert segs == [(0.5, 6.5), (8.5, 17.5), (28.5, 46.5), (48.5, 53.5)]


def test_editing_plan():
    plan = editing_plan(_report(), duration=60.0)
    assert plan["condensed"]["duration_s"] < plan["condensed"]["original_s"]
    assert plan["reel_points"] == sorted(plan["reel_points"])
    assert reel_segments({**plan, "points": _report()["points"]})[0][0] == 0.5


def test_cut_video(tmp_path):
    src = tmp_path / "src.mp4"
    fps, w, h = 30, 160, 90
    out = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(fps * 6):
        out.write(np.full((h, w, 3), i % 255, np.uint8))
    out.release()

    dst = cut_video(src, [(0.5, 1.5), (3.0, 5.0)], tmp_path / "cut.mp4")
    cap = cv2.VideoCapture(str(dst))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert abs(n - 3 * fps) <= 3
