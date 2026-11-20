import pytest

from tennis_vision.shots import miss_type, serve_zone, shot_direction


@pytest.mark.parametrize("x, zone", [(-0.3, "T"), (0.9, "T"), (-2.0, "body"), (3.4, "wide"), (-4.0, "wide")])
def test_serve_zone(x, zone):
    assert serve_zone(x) == zone


def test_miss_types():
    assert miss_type(0.5, -12.6, serve=False) == "long"
    assert miss_type(4.6, -8.0, serve=False) == "wide"
    assert miss_type(2.0, 7.3, serve=True) == "long"
    assert miss_type(1.0, 4.0, serve=True) == "wide"  # wrong side of the center line
    assert miss_type(0.5, -2.0, serve=True, own_half=True) == "net"


def test_shot_direction():
    assert shot_direction(3.0, -3.0) == "cross"
    assert shot_direction(-3.0, -3.0) == "line"
    assert shot_direction(3.0, 0.4) == "center"
    assert shot_direction(None, 3.0) is None


def test_stroke_wing():
    from tennis_vision.shots import stroke_wing

    # Near player faces the far end: ball at +x of the body is their right.
    assert stroke_wing(True, player_x=0.0, ball_x=0.8) == "forehand"
    assert stroke_wing(True, player_x=0.0, ball_x=-0.8) == "backhand"
    # Far player faces the camera, so their right is at -x.
    assert stroke_wing(False, player_x=0.0, ball_x=-0.8) == "forehand"
    assert stroke_wing(False, player_x=0.0, ball_x=0.8) == "backhand"
    assert stroke_wing(True, player_x=0.0, ball_x=0.8, handed="left") == "backhand"
    assert stroke_wing(True, player_x=0.0, ball_x=0.05) is None


def test_shot_type():
    from tennis_vision.shots import shot_type

    assert shot_type(0, False, None, None) == "serve"
    assert shot_type(2, True, 2.6, "forehand") == "overhead"
    assert shot_type(2, True, 1.2, "backhand") == "volley"
    assert shot_type(1, False, 0.9, "backhand") == "backhand"
    assert shot_type(1, False, None, None) == "stroke"
