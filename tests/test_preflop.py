from fractions import Fraction

import preflop
from cards import make_hand
from flop import FlopResult
from showdown import INCOMPATIBLE


def hand(*names: str) -> int:
    return make_hand(names)


FIRST = hand("Ah", "Kh")
SECOND = hand("Js", "Jd")


def test_fixed_hands_leave_17296_unordered_flops() -> None:
    flops = tuple(preflop.flop_children(FIRST, SECOND))

    assert len(flops) == 17296
    assert len(set(flops)) == len(flops)
    assert all(flop.bit_count() == 3 for flop in flops)
    assert not any(flop & (FIRST | SECOND) for flop in flops)


def test_preflop_aggregates_flop_results_recursively(monkeypatch) -> None:
    flops = (hand("2c", "3d", "4h"), hand("5c", "6d", "7h"))
    results = {
        flops[0]: FlopResult(flops[0], FIRST, SECOND, (), 10, 20, 2),
        flops[1]: FlopResult(flops[1], FIRST, SECOND, (), 7, 3, 1),
    }
    monkeypatch.setattr(preflop, "flop_children", lambda first, second: iter(flops))
    monkeypatch.setattr(
        preflop,
        "flop_outcomes",
        lambda flop, first, second: results[flop],
    )

    result = preflop.preflop_outcomes(FIRST, SECOND)

    assert len(result.flops) == 2
    assert (result.wins, result.losses, result.ties) == (17, 23, 3)
    assert result.total == 43
    assert result.equity == Fraction(37, 86)
    assert [child.flop for child in result.flops] == list(flops)


def test_overlapping_preflop_hands_are_incompatible() -> None:
    assert preflop.preflop_outcomes(FIRST, hand("Ah", "Qs")) == INCOMPATIBLE
    assert list(preflop.flop_children(FIRST, hand("Ah", "Qs"))) == []
