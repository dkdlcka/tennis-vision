import numpy as np

from synth import Camera, simulate, standard_match
from tennis_vision.court import DOUBLES_CORNERS, CourtHomography
from tennis_vision.events import find_rallies

FPS = 30


def _pixel_track(truth, cam, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    track = np.full((len(truth.positions), 2), np.nan)
    for f, p in enumerate(truth.positions):
        if p is None:
            continue
        q = cam.project([p])[0] + rng.normal(0, noise, 2)
        if 0 <= q[0] < cam.width and 0 <= q[1] < cam.height:
            track[f] = q
    return track


def _visible_bounces(truth, track):
    # Seen at the bounce or in the frame or two just before it.
    return [(f, x, y) for f, x, y in truth.bounces if (~np.isnan(track[max(0, f - 2) : f + 1, 0])).any()]


def _match(found, expected, tol_m):
    missing = []
    for f, x, y in expected:
        near = [b for b in found if abs(b.frame - f) <= 2 and np.hypot(b.court_xy[0] - x, b.court_xy[1] - y) < tol_m]
        if not near:
            missing.append((f, x, y))
    return missing


def test_bounces_and_hits_from_clean_track():
    cam = Camera()
    truth = simulate(standard_match(), FPS)
    track = _pixel_track(truth, cam)
    rallies = find_rallies(track, CourtHomography(cam.project(DOUBLES_CORNERS)), FPS, (cam.width, cam.height))
    assert len(rallies) == 4
    bounces = [b for r in rallies for b in r.bounces]
    expected = _visible_bounces(truth, track)
    # Landings as the ball leaves the picture are extrapolated, so allow more there.
    at_edge = [e for e in expected if np.isnan(track[min(e[0] + 2, len(track) - 1), 0])]
    assert not _match(bounces, [e for e in expected if e not in at_edge], 0.1)
    assert not _match(bounces, at_edge, 0.4)
    assert len(bounces) == len(expected)
    hits = [e for r in rallies for e in r.events if e.kind == "hit"]
    assert all(0.5 < h.height_m < 2.0 for h in hits)
    assert all(abs(h.court_xy[1]) > 10 for h in hits)  # players hit from near their baselines


def test_bounces_with_pixel_noise():
    cam = Camera()
    truth = simulate(standard_match(), FPS)
    track = _pixel_track(truth, cam, noise=0.7)
    rallies = find_rallies(track, CourtHomography(cam.project(DOUBLES_CORNERS)), FPS, (cam.width, cam.height))
    bounces = [b for r in rallies for b in r.bounces]
    expected = _visible_bounces(truth, track)
    at_edge = [e for e in expected if np.isnan(track[min(e[0] + 2, len(track) - 1), 0])]
    assert not _match(bounces, [e for e in expected if e not in at_edge], 0.35)
    assert not _match(bounces, at_edge, 0.8)


def test_serve_speed_is_recovered():
    cam = Camera()
    truth = simulate(standard_match()[-1:], FPS)
    track = _pixel_track(truth, cam)
    (rally,) = find_rallies(track, CourtHomography(cam.project(DOUBLES_CORNERS)), FPS, (cam.width, cam.height))
    first = next(i for i, p in enumerate(truth.positions) if p is not None)
    step = truth.positions[first + 1] - truth.positions[first]
    true_speed = np.linalg.norm(step[:2]) * FPS * 3.6  # horizontal speed is constant in flight
    assert abs(np.linalg.norm(rally.arcs[0].v0[:2]) * 3.6 - true_speed) < 3


def test_launch_speed_undoes_drag():
    from tennis_vision.events import DRAG_PER_M, Event, launch_speed_kmh

    # A 180 km/h serve from 2.7 m, flown with quadratic drag (and no gravity:
    # the estimate takes the path as straight), for 0.5 s.
    v0, dt, t, d = 50.0, 1e-4, 0.0, 0.0
    v = v0
    while t < 0.5:
        d += v * dt
        v -= DRAG_PER_M * v * v * dt
        t += dt
    ground = (d * d - 2.7 * 2.7) ** 0.5
    hit = Event("hit", 0.0, (0, 0), (0.0, -11.885))
    bounce = Event("bounce", 15.0, (0, 0), (0.0, -11.885 + ground))
    assert abs(launch_speed_kmh(hit, bounce, 30.0, 2.7) - v0 * 3.6) < 2.0
