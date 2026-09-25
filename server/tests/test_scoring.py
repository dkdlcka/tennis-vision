from tennis_vision.court import ServeBox, Side
from tennis_vision.scoring import Match, judge_point

DEUCE, AD = ServeBox.DEUCE, ServeBox.AD
NEAR, FAR = Side.NEAR, Side.FAR


def test_ace():
    r = judge_point([(-2.0, 4.0)], NEAR, DEUCE)
    assert (r.winner, r.reason) == (NEAR, "ace")


def test_serve_in_wrong_box_is_fault():
    r = judge_point([(2.0, 4.0)], NEAR, DEUCE)
    assert r.fault and r.winner is None


def test_serve_into_net_is_fault():
    r = judge_point([(0.5, -1.0)], NEAR, AD)
    assert r.fault


def test_rally_ends_with_ball_long():
    # serve in, return in, server hits past the far baseline
    r = judge_point([(-2.0, 4.0), (1.0, -8.0), (0.0, 12.5)], NEAR, DEUCE)
    assert (r.winner, r.reason) == (FAR, "out")
    assert [c.inside for c in r.calls] == [True, True, False]


def test_double_bounce_loses_for_that_side():
    r = judge_point([(-2.0, 4.0), (1.0, -3.0), (1.5, -9.0)], NEAR, DEUCE)
    assert (r.winner, r.reason) == (FAR, "double_bounce")


def test_ball_into_net_drops_on_own_side():
    # receiver's return hits the net and drops back on the far half
    r = judge_point([(-2.0, 4.0), (-1.0, 1.0)], NEAR, DEUCE)
    assert (r.winner, r.reason) == (NEAR, "double_bounce")


def test_unreturned_ball_is_a_winner():
    r = judge_point([(-2.0, 4.0), (1.0, -8.0)], NEAR, DEUCE)
    assert (r.winner, r.reason) == (FAR, "winner")


def _play(m: Match, seq: str) -> None:
    for p in seq:
        m.award(p)


def test_game_deuce_and_advantage():
    m = Match()
    _play(m, "AAABBB")
    assert m.display()["points"] == ["40", "40"]
    _play(m, "A")
    assert m.display()["points"] == ["AD", "40"]
    _play(m, "B")
    assert m.display()["points"] == ["40", "40"]
    _play(m, "AA")
    assert m.sets == [[1, 0]] and m.server == "B"


def test_no_ad_deciding_point():
    m = Match(no_ad=True)
    _play(m, "AAABBBB")
    assert m.sets == [[0, 1]]


def test_serve_box_alternates():
    m = Match()
    assert m.serve_box is DEUCE
    m.award("A")
    assert m.serve_box is AD


def test_ends_change_after_odd_games():
    m = Match()
    assert m.side_of_player("A") is NEAR
    _play(m, "AAAA")  # 1-0
    assert m.side_of_player("A") is FAR
    _play(m, "BBBB")  # 1-1
    assert m.side_of_player("A") is FAR
    _play(m, "AAAA")  # 2-1
    assert m.side_of_player("A") is NEAR


def test_set_and_tiebreak():
    m = Match(best_of=3)
    for _ in range(6):
        _play(m, "AAAA")
        _play(m, "BBBB")
    assert m.in_tiebreak
    first = m.current_server
    m.award("A")
    assert m.current_server != first  # one serve, then two each
    m.award("A")
    assert m.current_server != first
    m.award("A")
    assert m.current_server == first
    _play(m, "AAAA")  # 7-0 in the tiebreak
    assert m.sets == [[7, 6], [0, 0]]
    # the player who received first in the tiebreak serves the next set
    assert m.current_server != first


def test_double_fault():
    m = Match()
    assert m.record_fault() is False
    assert m.record_fault() is True
    assert m.points == [0, 1]


def test_match_winner():
    m = Match(best_of=3)
    for _ in range(12):
        _play(m, "AAAA")
    assert m.winner == "A"
