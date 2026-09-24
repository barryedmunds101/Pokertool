import random

import pytest

from cards import Deck, make_hand
from poker import HoldemDeal, deal_holdem, describe_deal


def test_six_player_deal_has_correct_sizes_and_no_overlap() -> None:
    deck = Deck(rng=random.Random(42))
    deal = deal_holdem(6, deck=deck)

    assert len(deal.private_hands) == 6
    assert all(hand.bit_count() == 2 for hand in deal.private_hands)
    assert deal.flop.bit_count() == 3
    assert deal.turn.bit_count() == deal.river.bit_count() == 1
    assert deal.board.bit_count() == 5
    assert deal.burn_cards.bit_count() == 3
    assert deal.all_dealt_cards.bit_count() == 20
    assert len(deck) == 32


def test_player_cards_combine_private_hand_and_board() -> None:
    deal = deal_holdem(2, deck=Deck(rng=random.Random(7)))
    assert deal.player_hand(0).bit_count() == 2
    assert deal.player_cards(0).bit_count() == 7
    with pytest.raises(IndexError):
        deal.player_hand(-1)
    with pytest.raises(IndexError):
        deal.player_hand(2)


def test_deal_without_burn_cards() -> None:
    deck = Deck(rng=random.Random(7))
    deal = deal_holdem(2, deck=deck, burn=False)
    assert deal.burn_cards == 0
    assert deal.all_dealt_cards.bit_count() == 9
    assert len(deck) == 43


@pytest.mark.parametrize("player_count", [1, 11])
def test_invalid_player_counts(player_count: int) -> None:
    with pytest.raises(ValueError):
        deal_holdem(player_count)


def test_duplicate_cards_in_manual_deal_are_rejected() -> None:
    duplicate = make_hand(["As", "Kd"])
    with pytest.raises(ValueError, match="same card"):
        HoldemDeal(
            (duplicate, duplicate),
            make_hand(["2c", "3c", "4c"]),
            make_hand(["5c"]),
            make_hand(["6c"]),
        )


def test_deal_description_contains_players_and_board() -> None:
    description = describe_deal(deal_holdem(2, deck=Deck(rng=random.Random(1))))
    assert "Player 1:" in description
    assert "Player 2:" in description
    assert "Board:" in description
