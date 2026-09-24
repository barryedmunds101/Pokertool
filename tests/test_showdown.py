import pytest

from cards import make_hand
from showdown import (
    INCOMPATIBLE,
    LOSS,
    PRIVATE_HANDS,
    TIE,
    WIN,
    best_hand_rank,
    compare_hands,
    compare_private_hands,
    comparison_matrix,
    comparison_vector,
)


def hand(*names: str) -> int:
    return make_hand(names)


def test_first_hand_wins_with_higher_pair() -> None:
    board = hand("2c", "7d", "9h", "Js", "3c")
    assert compare_private_hands(hand("Jc", "Ad"), hand("9c", "Kd"), board) == WIN


def test_compare_hands_is_the_public_comparison_function() -> None:
    board = hand("2c", "7d", "9h", "Js", "3c")
    assert compare_hands(hand("Jc", "Ad"), hand("9c", "Kd"), board) == WIN
    assert compare_private_hands is compare_hands


def test_ten_can_be_written_as_10() -> None:
    board = hand("2c", "7d", "9h", "Js", "3c")
    assert compare_hands(hand("10c", "Ad"), hand("9c", "Kd"), board) == LOSS


def test_comparison_vector_uses_all_private_hands_in_stable_order() -> None:
    board = hand("2c", "7d", "9h", "Js", "3c")
    first = hand("10c", "Ad")
    vector = comparison_vector(first, board)

    assert len(PRIVATE_HANDS) == 1326
    assert len(vector) == len(PRIVATE_HANDS)
    assert vector[PRIVATE_HANDS.index(hand("9c", "Kd"))] == LOSS
    assert vector[PRIVATE_HANDS.index(hand("10c", "Kd"))] == INCOMPATIBLE
    assert vector[PRIVATE_HANDS.index(hand("2c", "Kd"))] == INCOMPATIBLE


def test_comparison_matrix_has_matching_axes_and_perspectives() -> None:
    board = hand("2c", "7d", "9h", "Js", "3c")
    matrix = comparison_matrix(board)
    first_index = PRIVATE_HANDS.index(hand("10c", "Ad"))
    second_index = PRIVATE_HANDS.index(hand("9c", "Kd"))
    board_overlap_index = PRIVATE_HANDS.index(hand("2c", "Kh"))

    assert len(matrix) == len(PRIVATE_HANDS)
    assert all(len(row) == len(PRIVATE_HANDS) for row in matrix)
    assert matrix[first_index][second_index] == LOSS
    assert matrix[second_index][first_index] == WIN
    assert matrix[first_index][first_index] == INCOMPATIBLE
    assert all(value == INCOMPATIBLE for value in matrix[board_overlap_index])


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
    with pytest.raises(ValueError, match="board"):
        compare_private_hands(hand("2c", "Kd"), hand("Ac", "Qd"), board)


def test_incorrect_card_counts_are_rejected() -> None:
    board = hand("2c", "7d", "9h", "Js", "3c")
    with pytest.raises(ValueError, match="exactly 2"):
        compare_private_hands(hand("Ac"), hand("Kc", "Kd"), board)
    with pytest.raises(ValueError, match="exactly 5"):
        compare_private_hands(hand("Ac", "Ad"), hand("Kc", "Kd"), hand("2c", "3d"))
