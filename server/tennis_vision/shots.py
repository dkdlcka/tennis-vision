"""Describing shots: where a serve landed, which way a ball was hit, how it missed."""

from __future__ import annotations

from .court import HALF_DW, HALF_L, HALF_SW, SERVICE_LINE

CENTER_BAND = 1.37  # meters either side of the center line counted as "center"


def serve_zone(x: float) -> str:
    """Wide, body or T, by thirds of the service box measured from the center line."""
    d = abs(x)
    if d < HALF_SW / 3:
        return "T"
    if d > 2 * HALF_SW / 3:
        return "wide"
    return "body"


def miss_type(x: float, y: float, serve: bool, doubles: bool = False, own_half: bool = False) -> str:
    """How an out ball missed: 'net' when it dropped on the hitter's own half,
    else 'long' past the back line or 'wide' past a side line, whichever is larger.
    A serve inside the service line on the wrong side of the center line is 'wide'.
    """
    if own_half:
        return "net"
    long_by = abs(y) - (SERVICE_LINE if serve else HALF_L)
    wide_by = abs(x) - (HALF_SW if serve or not doubles else HALF_DW)
    if serve and long_by <= 0 and wide_by <= 0:
        return "wide"
    return "long" if long_by >= wide_by else "wide"


def shot_direction(contact_x: float | None, bounce_x: float) -> str | None:
    """Cross-court, down-the-line or center, from where the ball was struck and where it landed.

    Uses court x, which is the same for both players, so a ball struck on one side of
    the center line and landing on the other went cross-court.
    """
    if abs(bounce_x) < CENTER_BAND:
        return "center"
    if contact_x is None or abs(contact_x) < 0.5:
        return None
    return "line" if (contact_x > 0) == (bounce_x > 0) else "cross"


OVERHEAD_HEIGHT_M = 2.3  # a ball struck above this, not as a serve, is an overhead
WING_DEADBAND_M = 0.15  # contact this close to the body center is too close to call


def stroke_wing(side_near: bool, player_x: float, ball_x: float, handed: str = "right") -> str | None:
    """Forehand or backhand, from which side of the player's body the ball was struck.

    The near player faces the far end (+y), so their right hand is at +x; the far
    player faces the other way. A right-hander's forehand is on their right.
    """
    offset = ball_x - player_x
    if not side_near:
        offset = -offset
    if abs(offset) < WING_DEADBAND_M:
        return None
    on_right = offset > 0
    return "forehand" if on_right == (handed == "right") else "backhand"


def shot_type(index: int, volley: bool, height_m: float | None, wing: str | None) -> str:
    """serve, overhead, volley, forehand, backhand, or stroke when the wing is unknown."""
    if index == 0:
        return "serve"
    if height_m is not None and height_m > OVERHEAD_HEIGHT_M:
        return "overhead"
    if volley:
        return "volley"
    return wing or "stroke"
