"""Compatibility helpers for card bitsets.

These functions form the primitive layer for enumerating poker chance states.
Cards and card sets use the integer representation defined in :mod:`cards`.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import combinations

from cards import FULL_DECK, cards_in


def _validate_card_set(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer card bitset")
    if value < 0 or value & ~FULL_DECK:
        raise ValueError(f"{name} may only use bits 0 through 51")


def compatible(first: int, second: int) -> bool:
    """Return whether two card sets are disjoint."""
    _validate_card_set(first, "first")
    _validate_card_set(second, "second")
    return first & second == 0


def remaining_cards(dead_cards: int = 0) -> int:
    """Return every deck card not present in ``dead_cards``."""
    _validate_card_set(dead_cards, "dead_cards")
    return FULL_DECK & ~dead_cards


def compatible_private_hands(board: int) -> Iterator[int]:
    """Yield every two-card private hand compatible with ``board``.

    Each result is a two-bit integer.  Hands are yielded once each in stable
    deck-bit order, so a five-card board produces ``C(47, 2) == 1081`` hands.
    """
    _validate_card_set(board, "board")
    available_cards = cards_in(remaining_cards(board))
    for first, second in combinations(available_cards, 2):
        yield first | second
