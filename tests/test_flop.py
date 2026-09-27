from fractions import Fraction

from cards import make_hand
from flop import flop_outcomes, turn_children
from showdown import INCOMPATIBLE
from symmetry import permute_cards


def hand(*names: str) -> int:
    return make_hand(names)


FLOP = hand("2h", "7h", "Tc")
FIRST = hand("Ah", "Kh")
SECOND = hand("Js", "Jd")


def test_flop_has_49_children_and_fixed_hands_leave_45() -> None:
    assert len(turn_children(FLOP)) == 49
    children = turn_children(FLOP, FIRST | SECOND)
    assert len(children) == 45
    assert not any(card & (FLOP | FIRST | SECOND) for card, _ in children)


def test_flop_recursively_aggregates_all_turn_results() -> None:
    result = flop_outcomes(FLOP, FIRST, SECOND)

    assert len(result.turns) == 45
    assert result.total == 45 * 44 == 1980
    assert result.wins == sum(child.result.wins for child in result.turns)
    assert result.losses == sum(child.result.losses for child in result.turns)
    assert result.ties == sum(child.result.ties for child in result.turns)
    assert result.equity == Fraction(2 * result.wins + result.ties, 2 * result.total)
    assert all(len(child.turn_name) == 2 for child in result.turns)


def test_flop_hand_swap_and_suit_invariance() -> None:
    forward = flop_outcomes(FLOP, FIRST, SECOND)
    reverse = flop_outcomes(FLOP, SECOND, FIRST)
    permutation = (2, 0, 3, 1)
    permuted = flop_outcomes(
        permute_cards(FLOP, permutation),
        permute_cards(FIRST, permutation),
        permute_cards(SECOND, permutation),
    )

    assert (forward.wins, forward.losses, forward.ties) == (
        reverse.losses,
        reverse.wins,
        reverse.ties,
    )
    assert (forward.wins, forward.losses, forward.ties) == (
        permuted.wins,
        permuted.losses,
        permuted.ties,
    )


def test_incompatible_flop_returns_marker() -> None:
    assert flop_outcomes(FLOP, hand("Ah", "2h"), SECOND) == INCOMPATIBLE
    assert flop_outcomes(FLOP, FIRST, hand("Ah", "Qs")) == INCOMPATIBLE
