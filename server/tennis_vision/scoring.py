"""Tennis scoring and point adjudication.

`Match` keeps the score (points, games, sets, tiebreaks), who serves, from
which box, and which end each player is on. `judge_point` decides who won a
point from the ordered bounces of one rally.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .court import Side, ServeBox, playing_area, service_box, side_of

POINT_NAMES = ["0", "15", "30", "40"]


@dataclass
class Call:
    """One bounce with its line call."""

    x: float
    y: float
    side: Side
    inside: bool
    margin_m: float  # signed distance to the nearest relevant line, + is in
    kind: str  # "serve" or "rally"


@dataclass
class PointResult:
    winner: Side | None  # None when the point is replayed or undecided
    reason: str
    calls: list[Call]
    fault: bool = False  # serve fault, point continues with a second serve


def judge_point(
    bounces: list[tuple[float, float]],
    server_side: Side,
    box: ServeBox,
    doubles: bool = False,
) -> PointResult:
    """Adjudicate one serve and the rally that follows it.

    Rules applied, in order of the bounces:
    - The first bounce must land in the diagonal service box, else it is a fault.
    - After that, each shot must bounce in the opponent's half. A bounce out
      loses the point for the hitter.
    - Two bounces in a row on the same half (double bounce, or a shot into the
      net that drops back) lose the point for the player on that half.
    - If the ball stops being tracked after landing in on a half, the player on
      that half failed to return it.
    """
    calls: list[Call] = []
    if not bounces:
        return PointResult(None, "no_bounce", calls)

    x, y = bounces[0]
    target = service_box(server_side, box)
    margin = target.margin(x, y)
    serve_in = margin >= 0
    calls.append(Call(x, y, side_of(y), serve_in, margin, "serve"))
    if not serve_in:
        return PointResult(None, "fault", calls, fault=True)

    receiver = server_side.other
    last_side = receiver
    for x, y in bounces[1:]:
        side = side_of(y)
        area = playing_area(side, doubles)
        margin = area.margin(x, y)
        inside = margin >= 0
        if side is last_side:
            calls.append(Call(x, y, side, inside, margin, "rally"))
            return PointResult(side.other, "double_bounce", calls)
        calls.append(Call(x, y, side, inside, margin, "rally"))
        if not inside:
            # The player on last_side hit this ball out.
            return PointResult(side, "out", calls)
        last_side = side

    if len(bounces) == 1:
        return PointResult(server_side, "ace", calls)
    return PointResult(last_side.other, "winner", calls)


@dataclass
class Match:
    best_of: int = 3
    games_per_set: int = 6
    no_ad: bool = False
    first_server: str = "A"
    a_starts_near: bool = True

    sets: list[list[int]] = field(default_factory=lambda: [[0, 0]])
    points: list[int] = field(default_factory=lambda: [0, 0])
    server: str = ""
    second_serve: bool = False
    tiebreak_points_played: int = 0
    winner: str | None = None

    def __post_init__(self) -> None:
        self.server = self.server or self.first_server

    # ----- state helpers -----
    @property
    def in_tiebreak(self) -> bool:
        g = self.sets[-1]
        return g[0] == g[1] == self.games_per_set

    @property
    def games_played_total(self) -> int:
        return sum(a + b for a, b in self.sets)

    def side_of_player(self, player: str) -> Side:
        # Ends change after the first game and every two games after that,
        # i.e. after every odd total of games, and every six tiebreak points.
        swaps = (self.games_played_total + 1) // 2
        if self.in_tiebreak:
            swaps += self.tiebreak_points_played // 6
        a_near = self.a_starts_near ^ (swaps % 2 == 1)
        near_player = "A" if a_near else "B"
        return Side.NEAR if player == near_player else Side.FAR

    def player_on(self, side: Side) -> str:
        return "A" if self.side_of_player("A") is side else "B"

    @property
    def serve_box(self) -> ServeBox:
        played = self.tiebreak_points_played if self.in_tiebreak else sum(self.points)
        return ServeBox.DEUCE if played % 2 == 0 else ServeBox.AD

    @property
    def current_server(self) -> str:
        if not self.in_tiebreak:
            return self.server
        # In a tiebreak the player due to serve first serves one point, then two each.
        n = self.tiebreak_points_played
        switch = (n + 1) // 2 % 2 == 1
        return _other(self.server) if switch else self.server

    # ----- updates -----
    def record_fault(self) -> bool:
        """Returns True when this was a double fault (point lost)."""
        if self.second_serve:
            self.second_serve = False
            self.award(_other(self.current_server))
            return True
        self.second_serve = True
        return False

    def award(self, player: str) -> None:
        if self.winner:
            return
        self.second_serve = False
        i = 0 if player == "A" else 1
        self.points[i] += 1
        a, b = self.points
        if self.in_tiebreak:
            self.tiebreak_points_played += 1
            if max(a, b) >= 7 and abs(a - b) >= 2:
                self._win_game(i)
            return
        if self.no_ad and a == b == 3:
            return
        if self.no_ad and max(a, b) >= 4:
            self._win_game(i)
        elif max(a, b) >= 4 and abs(a - b) >= 2:
            self._win_game(i)

    def _win_game(self, i: int) -> None:
        was_tiebreak = self.in_tiebreak
        self.sets[-1][i] += 1
        self.points = [0, 0]
        if was_tiebreak:
            # The player who received first in the tiebreak serves the next game.
            self.tiebreak_points_played = 0
        else:
            self.server = _other(self.server)
        g = self.sets[-1]
        n = self.games_per_set
        set_won = (max(g) >= n and abs(g[0] - g[1]) >= 2) or max(g) == n + 1
        if set_won:
            if was_tiebreak:
                self.server = _other(self.server)
            won = sum(1 for s in self.sets if s[i] > s[1 - i])
            if won > self.best_of // 2:
                self.winner = "A" if i == 0 else "B"
            else:
                self.sets.append([0, 0])

    def display(self) -> dict:
        a, b = self.points
        if self.in_tiebreak or self.winner:
            pts = [str(a), str(b)]
        elif a >= 3 and b >= 3 and not self.no_ad:
            pts = ["40", "40"] if a == b else (["AD", "40"] if a > b else ["40", "AD"])
        else:
            pts = [POINT_NAMES[min(a, 3)], POINT_NAMES[min(b, 3)]]
        return {
            "sets": [list(s) for s in self.sets],
            "points": pts,
            "server": self.current_server,
            "second_serve": self.second_serve,
            "tiebreak": self.in_tiebreak,
            "winner": self.winner,
        }


def _other(p: str) -> str:
    return "B" if p == "A" else "A"
