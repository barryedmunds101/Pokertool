"""Playing cards represented as bits in a 52-bit integer.

Bits are numbered 0 through 51.  A single card has exactly one bit set and a
hand has one bit set for every card it contains.  This makes membership,
adding, and removing cards fast bitwise operations.
"""

from __future__ import annotations

import random
from collections.abc import Iterable


RANKS = "23456789TJQKA"
SUITS = "cdhs"  # clubs, diamonds, hearts, spades
FULL_DECK = (1 << 52) - 1


def card(rank: str, suit: str) -> int:
    """Return the one-bit integer for a card such as ``card("A", "s")``."""
    rank = rank.upper()
    suit = suit.lower()
    if rank not in RANKS or suit not in SUITS:
        raise ValueError(f"Invalid card: {rank}{suit}")
    return 1 << (SUITS.index(suit) * len(RANKS) + RANKS.index(rank))


def parse_card(value: str) -> int:
    """Convert two-character notation (for example ``'As'``) to a card bit."""
    value = value.strip()
    if len(value) != 2:
        raise ValueError(f"Card must contain a rank and suit: {value!r}")
    return card(value[0], value[1])


def card_name(value: int) -> str:
    """Convert a one-bit card integer to two-character notation."""
    if value <= 0 or value & (value - 1) or value & ~FULL_DECK:
        raise ValueError("A card must be exactly one bit within bits 0 through 51")
    index = value.bit_length() - 1
    return RANKS[index % len(RANKS)] + SUITS[index // len(RANKS)]


def make_hand(cards: Iterable[int | str]) -> int:
    """Build a hand bitset from card integers and/or card strings."""
    hand = 0
    for value in cards:
        value = parse_card(value) if isinstance(value, str) else value
        card_name(value)  # Validate the integer.
        if hand & value:
            raise ValueError(f"Duplicate card: {card_name(value)}")
        hand |= value
    return hand


def cards_in(hand: int) -> list[int]:
    """Return the individual card bits in a hand, ordered by bit position."""
    if hand < 0 or hand & ~FULL_DECK:
        raise ValueError("A hand may only use bits 0 through 51")
    return [1 << index for index in range(52) if hand & (1 << index)]


def hand_names(hand: int) -> list[str]:
    """Return readable names for every card in a hand."""
    return [card_name(value) for value in cards_in(hand)]


class Deck:
    """A shuffled deck whose cards use the 52-bit representation."""

    def __init__(self, *, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()
        self.reset()

    def reset(self, *, shuffle: bool = True) -> None:
        """Restore all 52 cards, shuffled by default."""
        self._cards = [1 << index for index in range(52)]
        if shuffle:
            self._rng.shuffle(self._cards)

    def __len__(self) -> int:
        return len(self._cards)

    def draw(self, count: int = 1) -> int:
        """Draw ``count`` cards and return them combined as one hand integer."""
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("count must be a positive integer")
        if count > len(self._cards):
            raise ValueError(f"Cannot draw {count} cards; only {len(self)} remain")

        hand = 0
        for _ in range(count):
            hand |= self._cards.pop()
        return hand


if __name__ == "__main__":
    deck = Deck()
    hand = deck.draw(5)
    print("Cards:", " ".join(hand_names(hand)))
    print("Integer:", hand)
    print("52-bit binary:", f"{hand:052b}")
