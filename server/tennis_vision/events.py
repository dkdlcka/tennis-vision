"""Turn a pixel ball track into rallies, bounces and hits.

Between two contacts (a racket or the ground) a ball flies a ballistic arc.
Knowing the camera, each stretch of the track is fitted with a 3D arc, and each
break between arcs is a contact. If the ball is at ground height there it was a
bounce, otherwise a hit. The arcs also give real ball speeds and how high the
ball cleared the net.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .court import HALF_DW, HALF_L, CourtCamera, CourtHomography

G = 9.81


@dataclass
class Arc:
    start: float  # frame
    end: float
    p0: np.ndarray  # position at `start`
    v0: np.ndarray  # velocity at `start`, m/s
    fps: float
    rms_px: float = 0.0

    def at(self, frame: float) -> np.ndarray:
        s = (frame - self.start) / self.fps
        return self.p0 + self.v0 * s + np.array([0, 0, -0.5 * G * s * s])

    def vel(self, frame: float) -> np.ndarray:
        s = (frame - self.start) / self.fps
        return self.v0 + np.array([0, 0, -G * s])

    def net_crossing(self) -> tuple[float, float] | None:
        """(frame, height) where the arc crosses the net plane inside this arc."""
        if abs(self.v0[1]) < 1e-6:
            return None
        s = -self.p0[1] / self.v0[1]
        frame = self.start + s * self.fps
        if not (self.start <= frame <= self.end):
            return None
        return frame, float(self.at(frame)[2])


@dataclass
class Event:
    kind: str  # "bounce" or "hit"
    frame: float  # sub-frame index of the contact
    image_xy: tuple[float, float]
    court_xy: tuple[float, float]
    height_m: float = 0.0
    speed_kmh: float | None = None  # incoming speed for bounces, outgoing for hits


@dataclass
class Rally:
    start_frame: int
    end_frame: int
    events: list[Event]
    arcs: list[Arc] = field(default_factory=list)

    @property
    def bounces(self) -> list[Event]:
        return [e for e in self.events if e.kind == "bounce"]

    def net_clearances(self) -> list[float]:
        out = []
        for a in self.arcs:
            c = a.net_crossing()
            if c is not None:
                out.append(round(c[1] - 0.914, 2))  # height above the net at its center
        return out


def split_rallies(
    track: np.ndarray, fps: float, max_gap_s: float = 1.2, min_len_s: float = 0.5
) -> list[tuple[int, int]]:
    """Frame ranges [start, end) where the ball is in play."""
    seen = np.where(~np.isnan(track[:, 0]))[0]
    if len(seen) == 0:
        return []
    max_gap = int(round(max_gap_s * fps))
    ranges = []
    start = prev = seen[0]
    for f in seen[1:]:
        if f - prev > max_gap:
            ranges.append((start, prev + 1))
            start = f
        prev = f
    ranges.append((start, prev + 1))
    return [(a, b) for a, b in ranges if (b - a) >= min_len_s * fps]


def fit_arc(camera: CourtCamera, frames: np.ndarray, uv: np.ndarray, fps: float) -> Arc:
    """Least-squares ballistic arc through image points.

    Each observation (u, v) of X(s) = p0 + v0 s - g s^2/2 z gives two equations
    that are linear in (p0, v0), so the fit is a single linear solve.
    """
    P = camera.P
    t0 = float(frames[0])
    rows, rhs = [], []
    for f, (u, v) in zip(frames, uv):
        s = (f - t0) / fps
        drop = np.array([0, 0, -0.5 * G * s * s, 1.0])
        for coord, pr in ((u, P[0]), (v, P[1])):
            a = coord * P[2] - pr  # a . [X, 1] = 0
            rows.append(np.concatenate([a[:3], a[:3] * s]))
            rhs.append(-a @ drop)
    A, b = np.array(rows), np.array(rhs)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    arc = Arc(t0, float(frames[-1]), sol[:3], sol[3:], fps)
    pred = camera.project(np.array([arc.at(f) for f in frames]))
    arc.rms_px = float(np.sqrt(np.mean(np.sum((pred - uv) ** 2, axis=1))))
    return arc


def fit_arcs_joint(
    camera: CourtCamera,
    frames: np.ndarray,
    uv: np.ndarray,
    pieces: list[tuple[int, int]],
    fps: float,
    contacts: list[float] | None = None,
) -> tuple[list[Arc], float]:
    """Fits consecutive arcs together so each one starts where the last ended.

    Short arcs seen from one camera are poorly constrained in depth; tying them
    to their neighbours at every contact removes most of that ambiguity. Rows
    are reweighted by depth once so every observation counts in pixels.
    """
    P = camera.P
    m = len(pieces)
    starts = [float(frames[a]) for a, _ in pieces]
    if contacts is None:
        contacts = [(frames[pieces[k][1]] + frames[pieces[k + 1][0]]) / 2 for k in range(m - 1)]
    weights = np.ones(len(uv))
    sol = None
    for _ in range(2):
        rows, rhs = [], []
        for k, (a, b) in enumerate(pieces):
            for i in range(a, b + 1):
                s = (frames[i] - starts[k]) / fps
                drop = np.array([0, 0, -0.5 * G * s * s, 1.0])
                for coord, pr in ((uv[i, 0], P[0]), (uv[i, 1], P[1])):
                    a_ = (coord * P[2] - pr) * weights[i]
                    row = np.zeros(6 * m)
                    row[6 * k : 6 * k + 3] = a_[:3]
                    row[6 * k + 3 : 6 * k + 6] = a_[:3] * s
                    rows.append(row)
                    rhs.append(-a_ @ drop)
        w_c = camera.f / 20
        for k, tc in enumerate(contacts):
            s1 = (tc - starts[k]) / fps
            s2 = (tc - starts[k + 1]) / fps
            for axis in range(3):
                row = np.zeros(6 * m)
                row[6 * k + axis] = w_c
                row[6 * k + 3 + axis] = w_c * s1
                row[6 * (k + 1) + axis] = -w_c
                row[6 * (k + 1) + 3 + axis] = -w_c * s2
                rows.append(row)
                rhs.append(w_c * 0.5 * G * (s1 * s1 - s2 * s2) if axis == 2 else 0.0)
        sol, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
        # Depth of every observation under the current solution.
        for k, (a, b) in enumerate(pieces):
            arc = Arc(starts[k], float(frames[b]), sol[6 * k : 6 * k + 3], sol[6 * k + 3 : 6 * k + 6], fps)
            for i in range(a, b + 1):
                x = np.append(arc.at(frames[i]), 1.0)
                weights[i] = 1.0 / max(float(P[2] @ x), 1e-3)
    arcs = []
    total = 0.0
    for k, (a, b) in enumerate(pieces):
        arc = Arc(starts[k], float(frames[b]), sol[6 * k : 6 * k + 3], sol[6 * k + 3 : 6 * k + 6], fps)
        pred = camera.project(np.array([arc.at(f) for f in frames[a : b + 1]]))
        err = np.sum((pred - uv[a : b + 1]) ** 2, axis=1)
        arc.rms_px = float(np.sqrt(np.mean(err)))
        total += float(err.sum())
        arcs.append(arc)
    # Breaks in continuity count too, in pixel-equivalent units.
    for k, tc in enumerate(contacts):
        gap = np.linalg.norm(arcs[k].at(tc) - arcs[k + 1].at(tc))
        total += (gap * camera.f / 20) ** 2
    return arcs, total


def _breakpoints(frames: np.ndarray, uv: np.ndarray, homography: CourtHomography, fps: float) -> list[int]:
    """Indices where the image velocity changes sharply (candidate contacts)."""
    if len(uv) < 7:
        return []
    court = homography.to_court(uv)
    ppm = np.linalg.norm(homography.to_image(court + [0.5, 0]) - homography.to_image(court - [0.5, 0]), axis=1)
    vel = np.gradient(uv, frames, axis=0) / np.maximum(ppm, 1e-6)[:, None] * fps
    w = 2
    change = np.zeros(len(uv))
    for i in range(w, len(uv) - w):
        change[i] = np.linalg.norm(vel[i + 1 : i + 1 + w].mean(0) - vel[i - w : i].mean(0))
    return [
        i for i in range(w, len(uv) - w) if change[i] > 2.5 and change[i] >= change[max(0, i - w) : i + w + 1].max()
    ]


def detect_events(
    track: np.ndarray,
    start: int,
    end: int,
    homography: CourtHomography,
    camera: CourtCamera,
    fps: float,
    max_hidden: int = 2,
) -> tuple[list[Event], list[Arc]]:
    """Events for one rally. Stretches where the ball was hidden for more than
    `max_hidden` frames may hide a contact, so each visible run is fitted alone."""
    seen = np.where(~np.isnan(track[start:end, 0]))[0] + start
    if len(seen) == 0:
        return [], []
    runs, run_start = [], seen[0]
    for prev, cur in zip(seen[:-1], seen[1:]):
        if cur - prev > max_hidden + 1:
            runs.append((run_start, prev + 1))
            run_start = cur
    runs.append((run_start, seen[-1] + 1))
    events: list[Event] = []
    arcs: list[Arc] = []
    for a, b in runs:
        ev, ar = _events_in_run(track, a, b, homography, camera, fps)
        events += ev
        arcs += ar
    return events, arcs


def _events_in_run(
    track: np.ndarray,
    start: int,
    end: int,
    homography: CourtHomography,
    camera: CourtCamera,
    fps: float,
) -> tuple[list[Event], list[Arc]]:
    seg = track[start:end]
    valid = ~np.isnan(seg[:, 0])
    frames = np.arange(start, end)[valid].astype(np.float64)
    uv = seg[valid]
    # Tracking glitches (a ghost, the shadow, a player's shoe) bend the arcs, so
    # drop points the fitted arcs disagree with strongly and segment again.
    for _ in range(3):
        if len(uv) < 6:
            return [], []
        cuts, contacts, arcs, pieces = _segment(frames, uv, homography, camera, fps)
        resid = np.concatenate(
            [
                np.linalg.norm(camera.project(np.array([arc.at(f) for f in frames[a : b + 1]])) - uv[a : b + 1], axis=1)
                for arc, (a, b) in zip(arcs, pieces)
            ]
        )
        limit = max(4.0, 4 * float(np.median(resid)))
        keep = resid <= limit
        if keep.all():
            break
        frames, uv = frames[keep], uv[keep]
    events = _classify(cuts, contacts, arcs, homography, camera)
    landing = _landing_at_end(arcs[-1], frames[-1], uv[-1], homography, camera, fps) if arcs else None
    if landing is not None:
        events.append(landing)
    return events, arcs


def _landing_at_end(
    arc: Arc,
    last_frame: float,
    last_uv: np.ndarray,
    homography: CourtHomography,
    camera: CourtCamera,
    fps: float,
) -> Event | None:
    """A bounce right where the ball leaves the picture (off the bottom edge, behind a player).

    With no arc after it there is no kink to see, so accept the landing when the
    last arc reaches the ground within a few frames of the final observation.
    """
    s_roots = np.roots([-0.5 * G, arc.v0[2], arc.p0[2]])
    ground = [arc.start + r.real * fps for r in s_roots if abs(r.imag) < 1e-9 and r.real > 0]
    ground = [f for f in ground if last_frame - 1.5 <= f <= last_frame + 3]
    if not ground or arc.vel(ground[0])[2] >= 0:
        return None
    f = max(ground[0], last_frame)  # the ball was still seen, so it had not landed yet
    # Extrapolate in the image, anchored on the last real observation, then map
    # to the ground plane; that stays accurate even when the arc's depth is off.
    offset = last_uv - camera.project(arc.at(last_frame)[None])[0]
    img = camera.project(arc.at(f)[None])[0] + offset
    p = homography.to_court(img[None])[0]
    speed = float(np.linalg.norm(arc.vel(f))) * 3.6
    return Event("bounce", float(f), (float(img[0]), float(img[1])), (float(p[0]), float(p[1])), 0.0, round(speed, 1))


def _segment(frames: np.ndarray, uv: np.ndarray, homography: CourtHomography, camera: CourtCamera, fps: float):

    # A cut c ends one piece at index c; the next piece starts at c + 1.
    # Start with every candidate break, then drop the ones a single arc explains.
    n = len(uv)
    cuts = [c for c in _breakpoints(frames, uv, homography, fps) if 2 <= c <= n - 4]
    tol = 1.5 + 0.002 * max(float(np.ptp(uv[:, 1])), 1.0)

    def fit(a: int, b: int) -> Arc:
        return fit_arc(camera, frames[a : b + 1], uv[a : b + 1], fps)

    def sse(a: int, b: int) -> float:
        arc = fit(a, b)
        return arc.rms_px**2 * (b - a + 1)

    def bounds(cs: list[int]) -> list[tuple[int, int]]:
        edges = [-1] + cs + [n - 1]
        return [(edges[k] + 1, edges[k + 1]) for k in range(len(edges) - 1)]

    while True:
        # Drop cuts that leave a piece too short to fit.
        pieces = bounds(cuts)
        short = [
            k for k in range(len(cuts)) if min(pieces[k][1] - pieces[k][0], pieces[k + 1][1] - pieces[k + 1][0]) < 3
        ]
        if short:
            cuts.pop(short[0])
            continue
        best = None
        for k in range(len(cuts)):
            a, b = pieces[k][0], pieces[k + 1][1]
            rms = fit(a, b).rms_px
            if rms < tol and (best is None or rms < best[0]):
                best = (rms, k)
        if best is None:
            break
        cuts.pop(best[1])

    # Place each contact precisely: slide its cut a couple of frames and try
    # fractional contact times, keeping whatever makes the joint fit best.
    def joint(cs: list[int], phases: list[float]):
        pcs = bounds(cs)
        tcs = [frames[c] + ph * (frames[c + 1] - frames[c]) for c, ph in zip(cs, phases)]
        return fit_arcs_joint(camera, frames, uv, pcs, fps, tcs)

    phases = [0.5] * len(cuts)
    for _ in range(2):
        for k in range(len(cuts)):
            lo = (cuts[k - 1] if k > 0 else -1) + 4
            hi = (cuts[k + 1] if k + 1 < len(cuts) else n - 1) - 4
            best = None
            for c in range(max(lo, cuts[k] - 2), min(hi, cuts[k] + 2) + 1):
                for ph in (0.1, 0.3, 0.5, 0.7, 0.9):
                    trial_c, trial_p = cuts.copy(), phases.copy()
                    trial_c[k], trial_p[k] = c, ph
                    cost = joint(trial_c, trial_p)[1]
                    if best is None or cost < best[0]:
                        best = (cost, c, ph)
            if best is not None:
                cuts[k], phases[k] = best[1], best[2]

    pieces = bounds(cuts)
    contacts = [frames[c] + ph * (frames[c + 1] - frames[c]) for c, ph in zip(cuts, phases)]
    arcs, _ = fit_arcs_joint(camera, frames, uv, pieces, fps, contacts)
    return cuts, contacts, arcs, pieces


def _classify(cuts, contacts, arcs: list[Arc], homography: CourtHomography, camera: CourtCamera) -> list[Event]:
    events: list[Event] = []
    for k in range(len(cuts)):
        before, after = arcs[k], arcs[k + 1]
        f_contact = float(contacts[k])
        contact = (before.at(f_contact) + after.at(f_contact)) / 2
        vy_before, vy_after = before.vel(f_contact)[1], after.vel(f_contact)[1]
        # A hit sends the ball back toward the other end; a bounce keeps it going.
        vz_before, vz_after = before.vel(f_contact)[2], after.vel(f_contact)[2]
        is_bounce = bool(
            np.sign(vy_before) == np.sign(vy_after)
            and min(abs(vy_before), abs(vy_after)) > 1.0
            and vz_before < 0 < vz_after
        )
        if not is_bounce and np.sign(vy_before) == np.sign(vy_after):
            continue  # a kink that is neither a clean bounce nor a return; not a contact we can name
        img = camera.project(contact[None])[0]
        if is_bounce:
            # On the ground, the image point maps exactly onto the court plane.
            ground = homography.to_court(img[None])[0]
            court_xy = (float(ground[0]), float(ground[1]))
            speed = float(np.linalg.norm(before.vel(f_contact))) * 3.6
            height = 0.0
        else:
            court_xy = (float(contact[0]), float(contact[1]))
            speed = float(np.linalg.norm(after.vel(f_contact))) * 3.6
            height = float(contact[2])
        events.append(
            Event(
                "bounce" if is_bounce else "hit",
                f_contact,
                (float(img[0]), float(img[1])),
                court_xy,
                height,
                round(speed, 1),
            )
        )
    return events


def remove_spikes(track: np.ndarray, tol_px: float, span: int = 4) -> np.ndarray:
    """Drops points that neither side of the track agrees with.

    Each point is predicted by a parabola through the points just before it and
    one through the points just after it. The tracker jumping onto a nearby
    blob for a frame or two misses both; a real contact (a bounce or a hit)
    still continues at least one side.
    """
    out = track.copy()
    seen = ~np.isnan(track[:, 0])
    for f in np.where(seen)[0]:
        errors = []
        for side in (np.arange(f - span, f), np.arange(f + 1, f + span + 1)):
            side = side[(side >= 0) & (side < len(track))]
            side = side[seen[side]]
            if len(side) < 3:
                break
            fit = [np.polyval(np.polyfit(side - f, track[side, k], 2), 0.0) for k in (0, 1)]
            errors.append(np.hypot(fit[0] - track[f, 0], fit[1] - track[f, 1]))
        if len(errors) == 2 and min(errors) > tol_px:
            out[f] = np.nan
    return out


def find_rallies(
    track: np.ndarray,
    homography: CourtHomography,
    fps: float,
    image_size: tuple[int, int],
    method: str = "3d",
) -> list[Rally]:
    """Rallies with their bounces and hits.

    "3d" fits ballistic arcs through a camera recovered from the court, which
    also gives speeds and net clearance. "2d" reads contacts from the image
    track alone (`detect_events_2d`), for a camera looking straight down the
    court, where that recovery is ill-conditioned.
    """
    camera = CourtCamera(homography, image_size)
    track = remove_spikes(track, tol_px=max(6.0, 8.0 * image_size[0] / 1280))
    rallies = []
    for a, b in split_rallies(track, fps):
        if method == "2d":
            rallies.append(Rally(a, b, detect_events_2d(track, a, b, homography, 1.5 * image_size[0] / 1280)))
        else:
            events, arcs = detect_events(track, a, b, homography, camera, fps)
            rallies.append(Rally(a, b, events, arcs))
    return rallies


def _quad_sse(t: np.ndarray, uv: np.ndarray) -> float:
    if len(t) < 4:
        return 0.0
    tc = t - t.mean()
    a = np.stack([np.ones_like(tc), tc, tc * tc], axis=1)
    sol, res, *_ = np.linalg.lstsq(a, uv, rcond=None)
    return float(np.sum((a @ sol - uv) ** 2))


def segment_2d(frames: np.ndarray, uv: np.ndarray, noise_px: float, min_len: int = 4, max_len: int = 90) -> list[int]:
    """Splits an image track into smooth pieces (quadratic in time in both coordinates).

    Optimal partition by dynamic programming: each piece costs its squared
    residual plus a fixed price, so a break is only worth it where the path
    really kinks. Returns the index where each piece after the first starts.
    """
    n = len(uv)
    price = 30.0 * noise_px * noise_px
    best = np.full(n + 1, np.inf)
    best[0] = 0.0
    back = np.zeros(n + 1, int)
    for j in range(min_len, n + 1):
        for i in range(max(0, j - max_len), j - min_len + 1):
            if not np.isfinite(best[i]):
                continue
            c = best[i] + price + _quad_sse(frames[i:j], uv[i:j])
            if c < best[j]:
                best[j], back[j] = c, i
    if not np.isfinite(best[n]):
        return []
    starts, j = [], n
    while j > 0:
        j = back[j]
        if j > 0:
            starts.append(j)
    return sorted(starts)


def _piece_velocity(t: np.ndarray, uv: np.ndarray, at: float) -> np.ndarray:
    deg = 2 if len(t) >= 5 else 1
    return np.array([np.polyval(np.polyder(np.polyfit(t - at, uv[:, k], deg)), 0.0) for k in (0, 1)])


@dataclass
class _Kink:
    frame: float
    img: np.ndarray
    ground: np.ndarray
    v_in: np.ndarray
    v_out: np.ndarray
    drift_out: float  # image y travel over the piece after the kink (+ is down the screen)


def detect_events_2d(
    track: np.ndarray, start: int, end: int, homography: CourtHomography, noise_px: float = 1.5
) -> list[Event]:
    """Bounces and hits from the image track alone, for a camera behind the near baseline.

    Needs no 3D camera, which a camera looking straight down the court cannot
    give reliably. The track is split into smooth pieces; each kink is a contact.

    Far from the camera the ball's image motion is mostly its rise and fall: a
    kink from falling to rising is a bounce, one from rising to falling (the
    ball turned back toward the camera) is the far player's shot. Near the
    camera, coming closer and falling both move the ball down the screen, so
    near-side kinks are read from the order of play instead: the ball bounces,
    then the near player hits it. The first kink is the serve.

    A bounce happens on the ground, so its image point maps straight onto the
    court. Racket contacts are placed at the hitter's baseline.
    """
    seg = track[start:end]
    ok = ~np.isnan(seg[:, 0])
    frames = np.arange(start, end)[ok].astype(np.float64)
    uv = seg[ok]
    if len(uv) < 8:
        return []
    cuts = segment_2d(frames, uv, noise_px)
    bounds = [0, *cuts, len(uv)]
    kinks: list[_Kink] = []
    for k, c in enumerate(cuts):
        a, b = bounds[k], bounds[k + 2]
        tb, ta = frames[a:c], frames[c:b]
        tc = (frames[c - 1] + frames[c]) / 2
        v_in = _piece_velocity(tb, uv[a:c], tc)
        v_out = _piece_velocity(ta, uv[c:b], tc)
        if np.hypot(*(v_out - v_in)) < 0.25 * max(np.hypot(*v_in), np.hypot(*v_out)):
            continue  # a gentle bend, not a contact
        p_in = [np.polyval(np.polyfit(tb - tc, uv[a:c, j], min(2, len(tb) - 1)), 0.0) for j in (0, 1)]
        p_out = [np.polyval(np.polyfit(ta - tc, uv[c:b, j], min(2, len(ta) - 1)), 0.0) for j in (0, 1)]
        img = (np.array(p_in) + np.array(p_out)) / 2
        ground = homography.to_court(img[None])[0]
        if abs(ground[0]) > HALF_DW + 1.5:
            continue  # off to the side of the court: the tracker was on something else
        kinks.append(_Kink(float(tc), img, ground, v_in, v_out, float(uv[b - 1, 1] - uv[c, 1])))
    if not kinks:
        return []

    events: list[Event] = []

    def add(kind: str, kink: _Kink, far: bool) -> None:
        if kind == "bounce":
            if abs(kink.ground[1]) > HALF_L + 4:
                return  # no ball lands that far back; the tracker was on something else
            xy = (float(kink.ground[0]), float(kink.ground[1]))
        else:
            xy = (float(kink.ground[0]), HALF_L if far else -HALF_L)
        events.append(Event(kind, kink.frame, (float(kink.img[0]), float(kink.img[1])), xy))

    # A serve seen from the toss: a slow, mostly vertical piece, then the ball
    # leaves fast; it travels away (up the screen) from a near server.
    first = kinks[0]
    tossed = np.hypot(*first.v_in) < 0.2 * np.hypot(*first.v_out)
    if tossed:
        add("hit", first, far=first.drift_out > 0)
        kinks = kinks[1:]
    near_group: list[_Kink] = []

    def flush(next_is_far_bounce: bool, last: bool) -> None:
        if len(near_group) >= 2:
            add("bounce", near_group[0], far=False)
            add("hit", near_group[-1], far=False)
        elif near_group:
            # One near-side kink: a volley if the ball next lands on the far
            # side, the ball's final landing if nothing follows.
            k = near_group[0]
            add("hit" if next_is_far_bounce or (not last and k.v_out[1] < 0) else "bounce", k, far=False)
        near_group.clear()

    for k in kinks:
        if k.ground[1] < 0.5:
            near_group.append(k)
            continue
        falling_to_rising = k.v_in[1] > 0 and k.v_out[1] < 0
        rising_to_falling = k.v_in[1] < 0 and k.v_out[1] > 0
        far_bounce = falling_to_rising and k.ground[1] < HALF_L + 4
        flush(next_is_far_bounce=far_bounce, last=False)
        if far_bounce:
            add("bounce", k, far=True)
        elif rising_to_falling or k.ground[1] >= HALF_L + 4:
            add("hit", k, far=True)
    flush(next_is_far_bounce=False, last=True)
    return _drop_impossible(events)


def _drop_impossible(events: list[Event]) -> list[Event]:
    """A bounce must land on the side opposite the last hitter (or where the ball already
    bounced, for a second bounce); anything else is a misread kink."""
    out: list[Event] = []
    last_hit_side = None
    for e in events:
        side = "near" if e.court_xy[1] < 0 else "far"
        if e.kind == "hit":
            if out and out[-1].kind == "hit" and ("near" if out[-1].court_xy[1] < 0 else "far") == side:
                continue  # the same player cannot hit twice in a row
            last_hit_side = side
        elif last_hit_side == side:
            continue
        out.append(e)
    return out
