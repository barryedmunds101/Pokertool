"""The fixed ambient space of unordered two-card private hands."""

from __future__ import annotations

from itertools import combinations

from cards import FULL_DECK, cards_in, hand_names
from compatibility import _validate_card_set


PRIVATE_HANDS = tuple(
    first | second for first, second in combinations(cards_in(FULL_DECK), 2)
)
"""All 1,326 private hands in stable deck-bit order."""

HAND_INDEX = {hand: index for index, hand in enumerate(PRIVATE_HANDS)}
"""Map a two-card bitset to its coordinate in :data:`PRIVATE_HANDS`."""


def hand_label(hand: int) -> str:
    """Return a stable, human-readable label such as ``'2c3c'``."""
    try:
        HAND_INDEX[hand]
    except (KeyError, TypeError):
        raise ValueError("hand must be a two-card private hand") from None
    return "".join(hand_names(hand))


def compatible_hand_indices(board: int) -> tuple[int, ...]:
    """Return ambient coordinates of private hands disjoint from ``board``."""
    _validate_card_set(board, "board")
    return tuple(index for index, hand in enumerate(PRIVATE_HANDS) if not hand & board)
