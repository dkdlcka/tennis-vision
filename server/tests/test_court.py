import numpy as np
import pytest

from synth import Camera
from tennis_vision.court import (
    DOUBLES_CORNERS,
    HALF_L,
    HALF_SW,
    SERVICE_LINE,
    CourtCamera,
    CourtHomography,
    ServeBox,
    Side,
    playing_area,
    service_box,
)


def test_line_counts_as_in():
    area = playing_area(Side.FAR)
    assert area.margin(HALF_SW, 5.0) == pytest.approx(0.0)
    assert area.margin(HALF_SW + 0.01, 5.0) == pytest.approx(-0.01)
    assert area.margin(0.0, HALF_L - 0.5) == pytest.approx(0.5)


def test_doubles_alleys_only_count_in_doubles():
    assert playing_area(Side.NEAR).margin(HALF_SW + 0.5, -5) < 0
    assert playing_area(Side.NEAR, doubles=True).margin(HALF_SW + 0.5, -5) > 0


@pytest.mark.parametrize(
    "server, box, point, inside",
    [
        (Side.NEAR, ServeBox.DEUCE, (-2.0, 4.0), True),  # diagonal: near server, deuce -> far left box
        (Side.NEAR, ServeBox.DEUCE, (2.0, 4.0), False),
        (Side.NEAR, ServeBox.AD, (2.0, 4.0), True),
        (Side.FAR, ServeBox.DEUCE, (2.0, -4.0), True),
        (Side.FAR, ServeBox.AD, (-2.0, -4.0), True),
        (Side.NEAR, ServeBox.DEUCE, (-2.0, SERVICE_LINE + 0.05), False),
    ],
)
def test_service_boxes(server, box, point, inside):
    assert (service_box(server, box).margin(*point) >= 0) is inside


def test_homography_round_trip():
    cam = Camera()
    h = CourtHomography(cam.project(DOUBLES_CORNERS))
    pts = np.array([[1.0, 2.0], [-3.0, -8.0], [4.1, 11.0]])
    assert np.allclose(h.to_court(h.to_image(pts)), pts)
    assert np.allclose(h.to_image(pts), cam.project(pts), atol=1e-6)


def test_camera_recovered_from_court():
    cam = Camera()
    recovered = CourtCamera(CourtHomography(cam.project(DOUBLES_CORNERS)), (cam.width, cam.height))
    assert recovered.f == pytest.approx(cam.f, rel=1e-4)
    assert np.allclose(recovered.center, cam.c, atol=1e-3)
    pts = np.array([[1.0, 2.0, 3.0], [-3.0, -8.0, 1.5]])
    assert np.allclose(recovered.project(pts), cam.project(pts), atol=1e-3)
