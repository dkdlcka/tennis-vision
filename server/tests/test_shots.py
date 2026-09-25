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
