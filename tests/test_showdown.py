import pytest

from cards import make_hand
from showdown import LOSS, TIE, WIN, best_hand_rank, compare_private_hands


def hand(*names: str) -> int:
    return make_hand(names)


def test_first_hand_wins_with_higher_pair() -> None:
    board = hand("2c", "7d", "9h", "Js", "3c")
    assert compare_private_hands(hand("Jc", "Ad"), hand("9c", "Kd"), board) == WIN


def test_first_hand_loses_to_flush() -> None:
    board = hand("2h", "5h", "9h", "Ks", "3c")
    assert compare_private_hands(hand("Kc", "Kd"), hand("Ah", "Jh"), board) == LOSS


def test_board_can_force_a_tie() -> None:
    board = hand("Ah", "Kh", "Qh", "Jh", "Th")
    assert compare_private_hands(hand("2c", "3d"), hand("4c", "5d"), board) == TIE


def test_wheel_straight_is_recognised() -> None:
    wheel = best_hand_rank(hand("Ac", "2d", "3h", "4s", "5c", "Kd", "Qd"))
    six_high = best_hand_rank(hand("2c", "3d", "4h", "5s", "6c", "Kd", "Qd"))
    assert six_high > wheel


def test_best_five_cards_are_selected_from_seven() -> None:
    straight_flush = best_hand_rank(hand("5s", "6s", "7s", "8s", "9s", "Ah", "Ad"))
    four_of_a_kind = best_hand_rank(hand("Kc", "Kd", "Kh", "Ks", "2c", "3d", "4h"))
    assert straight_flush > four_of_a_kind


def test_overlapping_cards_are_rejected() -> None:
    board = hand("2c", "7d", "9h", "Js", "3c")
    with pytest.raises(ValueError, match="private hands overlap"):
        compare_private_hands(hand("Ac", "Kd"), hand("Ac", "Qd"), board)
    with pytest.raises(ValueError, match="river board"):
        compare_private_hands(hand("2c", "Kd"), hand("Ac", "Qd"), board)


def test_incorrect_card_counts_are_rejected() -> None:
    board = hand("2c", "7d", "9h", "Js", "3c")
    with pytest.raises(ValueError, match="exactly 2"):
        compare_private_hands(hand("Ac"), hand("Kc", "Kd"), board)
    with pytest.raises(ValueError, match="exactly 5"):
        compare_private_hands(hand("Ac", "Ad"), hand("Kc", "Kd"), hand("2c", "3d"))
