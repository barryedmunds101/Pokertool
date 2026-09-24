from math import comb

import pytest

from cards import FULL_DECK, make_hand, parse_card
from compatibility import compatible, compatible_private_hands, remaining_cards


def test_compatible_card_sets() -> None:
    aces = make_hand(["Ac", "Ad", "Ah", "As"])
    kings = make_hand(["Kc", "Kd", "Kh", "Ks"])

    assert compatible(aces, kings)
    assert not compatible(aces, make_hand(["As", "Kd"]))


def test_remaining_cards_is_deck_complement() -> None:
    dead = make_hand(["As", "Kd", "2c"])
    available = remaining_cards(dead)

    assert available == FULL_DECK & ~dead
    assert available.bit_count() == 49
    assert available & dead == 0
    assert available | dead == FULL_DECK


def test_private_hands_compatible_with_five_card_board() -> None:
    board = make_hand(["As", "Kd", "Qh", "Jc", "2s"])
    hands = list(compatible_private_hands(board))

    assert len(hands) == comb(47, 2) == 1081
    assert len(set(hands)) == len(hands)
    assert all(hand.bit_count() == 2 for hand in hands)
    assert all(compatible(hand, board) for hand in hands)


def test_empty_board_generates_all_starting_card_combinations() -> None:
    assert sum(1 for _ in compatible_private_hands(0)) == comb(52, 2)


@pytest.mark.parametrize("invalid", [-1, 1 << 52, True, "As"])
def test_invalid_card_sets_are_rejected(invalid: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        remaining_cards(invalid)  # type: ignore[arg-type]


def test_specific_private_hand_is_excluded_by_board_overlap() -> None:
    board = make_hand(["As", "Kd", "Qh", "Jc", "2s"])
    hands = set(compatible_private_hands(board))
    assert make_hand(["Ac", "Ad"]) in hands
    assert make_hand(["As", "Ad"]) not in hands
    assert parse_card("As") & board
