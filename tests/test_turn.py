from fractions import Fraction

from cards import make_hand
from showdown import INCOMPATIBLE, LOSS, TIE, WIN
from symmetry import permute_cards
from turn import river_children, turn_outcomes


def hand(*names: str) -> int:
    return make_hand(names)


TURN = hand("2h", "7h", "Tc", "Jc")
FIRST = hand("Ah", "Kh")
SECOND = hand("Js", "Jd")


def test_turn_has_48_children_and_fixed_hands_leave_44() -> None:
    assert len(river_children(TURN)) == 48
    children = river_children(TURN, FIRST | SECOND)
    assert len(children) == 44
    assert not any(river & (TURN | FIRST | SECOND) for river, _ in children)
    assert all(board == TURN | river and board.bit_count() == 5 for river, board in children)


def test_outcomes_are_labelled_and_counts_are_exact() -> None:
    result = turn_outcomes(TURN, FIRST, SECOND)

    assert len(result.rivers) == result.total == 44
    assert result.wins + result.losses + result.ties == 44
    assert {item.outcome for item in result.rivers} <= {"win", "loss", "tie"}
    assert all(len(item.river_name) == 2 for item in result.rivers)
    assert result.equity == Fraction(2 * result.wins + result.ties, 88)


def test_swapping_hands_swaps_wins_and_losses() -> None:
    forward = turn_outcomes(TURN, FIRST, SECOND)
    reverse = turn_outcomes(TURN, SECOND, FIRST)

    assert (forward.wins, forward.losses, forward.ties) == (
        reverse.losses,
        reverse.wins,
        reverse.ties,
    )
    assert [item.river for item in forward.rivers] == [item.river for item in reverse.rivers]
    assert all(
        (left.result, right.result) in {(WIN, LOSS), (LOSS, WIN), (TIE, TIE)}
        for left, right in zip(forward.rivers, reverse.rivers)
    )


def test_suit_permutation_preserves_outcome_counts() -> None:
    permutation = (2, 0, 3, 1)
    original = turn_outcomes(TURN, FIRST, SECOND)
    permuted = turn_outcomes(
        permute_cards(TURN, permutation),
        permute_cards(FIRST, permutation),
        permute_cards(SECOND, permutation),
    )

    assert (original.wins, original.losses, original.ties) == (
        permuted.wins,
        permuted.losses,
        permuted.ties,
    )


def test_overlapping_cards_are_incompatible() -> None:
    assert turn_outcomes(TURN, hand("Ah", "2h"), SECOND) == INCOMPATIBLE
    assert turn_outcomes(TURN, FIRST, hand("Ah", "Qs")) == INCOMPATIBLE
