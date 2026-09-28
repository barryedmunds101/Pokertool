from fractions import Fraction

import preflop
from cards import make_hand
from chance import ChanceState, canonical_chance_children
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
    monkeypatch.setattr(
        preflop,
        "preflop_flop_outcomes",
        lambda first, second: (
            preflop.FlopOutcome.from_result(results[flop]) for flop in flops
        ),
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


def test_preflop_keeps_all_flops_while_calculating_one_per_suit_orbit(
    monkeypatch,
) -> None:
    first = hand("Ac", "Ad")
    second = hand("Kc", "Kd")
    canonical_count = len(canonical_chance_children(ChanceState(first=first, second=second)))
    calls = []

    def fake_flop_outcomes(flop, first_hand, second_hand):
        calls.append(flop)
        return FlopResult(flop, first_hand, second_hand, (), 1, 0, 0)

    monkeypatch.setattr(preflop, "flop_outcomes", fake_flop_outcomes)
    summaries = tuple(preflop.preflop_flop_outcomes(first, second))

    assert len(summaries) == 17296
    assert len({summary.flop for summary in summaries}) == 17296
    assert len(calls) == canonical_count < len(summaries)
