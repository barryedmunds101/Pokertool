import random

import pytest

from cards import (
    Deck,
    FULL_DECK,
    card,
    card_name,
    cards_in,
    hand_names,
    make_hand,
    parse_card,
)


def test_card_round_trip() -> None:
    assert parse_card("As") == card("A", "s") == 1 << 51
    assert card_name(parse_card("As")) == "As"
    assert parse_card("2c") == 1


def test_hand_representation() -> None:
    names = {"As", "Kd", "Th", "2c", "7s"}
    hand = make_hand(names)

    assert hand.bit_count() == 5
    assert set(hand_names(hand)) == names
    assert len(cards_in(hand)) == 5
    assert hand & parse_card("As")
    assert not hand & parse_card("Ah")


def test_deck_draws_unique_cards_and_resets() -> None:
    deck = Deck(rng=random.Random(42))
    first = deck.draw(5)
    second = deck.draw(5)

    assert first.bit_count() == second.bit_count() == 5
    assert first & second == 0
    assert len(deck) == 42

    deck.reset(shuffle=False)
    assert deck.draw(52) == FULL_DECK
    assert len(deck) == 0


@pytest.mark.parametrize("value", ["1s", "ace of spades", "", "Asx"])
def test_invalid_card_names_are_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        parse_card(value)


def test_invalid_hands_and_draws_are_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        make_hand(["As", "As"])
    with pytest.raises(ValueError):
        card_name(0)
    with pytest.raises(ValueError):
        Deck().draw(53)
