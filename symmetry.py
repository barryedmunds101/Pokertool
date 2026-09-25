"""The action of the 24 suit permutations on cards, hands, and boards."""

from __future__ import annotations

from itertools import permutations

import numpy as np

from cards import RANKS
from compatibility import _validate_card_set
from hand_space import HAND_INDEX, PRIVATE_HANDS


SuitPermutation = tuple[int, int, int, int]
SUIT_PERMUTATIONS: tuple[SuitPermutation, ...] = tuple(permutations(range(4)))


def permute_card(card: int, permutation: SuitPermutation) -> int:
    """Apply a suit permutation to a one-card bitset."""
    _validate_permutation(permutation)
    _validate_card_set(card, "card")
    if card.bit_count() != 1:
        raise ValueError("card must contain exactly one card")
    index = card.bit_length() - 1
    return 1 << (permutation[index // 13] * 13 + index % 13)


def permute_cards(cards: int, permutation: SuitPermutation) -> int:
    """Apply a suit permutation to an arbitrary card bitset."""
    _validate_permutation(permutation)
    _validate_card_set(cards, "cards")
    result = 0
    suit_mask = (1 << len(RANKS)) - 1
    for old_suit, new_suit in enumerate(permutation):
        block = (cards >> (old_suit * 13)) & suit_mask
        result |= block << (new_suit * 13)
    return result


def _validate_permutation(permutation: SuitPermutation) -> None:
    if tuple(sorted(permutation)) != (0, 1, 2, 3):
        raise ValueError("permutation must contain each suit index exactly once")


def compose(first: SuitPermutation, second: SuitPermutation) -> SuitPermutation:
    """Return ``first`` after ``second`` (the action ``first(second(x))``)."""
    _validate_permutation(first)
    _validate_permutation(second)
    return tuple(first[second[suit]] for suit in range(4))  # type: ignore[return-value]


def inverse(permutation: SuitPermutation) -> SuitPermutation:
    """Return the inverse suit permutation."""
    _validate_permutation(permutation)
    result = [0] * 4
    for old, new in enumerate(permutation):
        result[new] = old
    return tuple(result)  # type: ignore[return-value]


def hand_permutation(permutation: SuitPermutation) -> np.ndarray:
    """Return the ambient index map ``i -> g(i)`` as an integer array."""
    return np.fromiter(
        (HAND_INDEX[permute_cards(hand, permutation)] for hand in PRIVATE_HANDS),
        dtype=np.int16,
        count=len(PRIVATE_HANDS),
    )


HAND_PERMUTATIONS = tuple(hand_permutation(g) for g in SUIT_PERMUTATIONS)


def canonical_board(board: int) -> int:
    """Return the least integer in the board's suit-permutation orbit."""
    _validate_card_set(board, "board")
    return min(permute_cards(board, g) for g in SUIT_PERMUTATIONS)


def stabilizer(board: int) -> tuple[SuitPermutation, ...]:
    """Return suit permutations that leave ``board`` unchanged."""
    _validate_card_set(board, "board")
    return tuple(g for g in SUIT_PERMUTATIONS if permute_cards(board, g) == board)


def private_hand_orbits() -> tuple[tuple[int, ...], ...]:
    """Return the 169 suit orbits as tuples of ambient hand indices."""
    unseen = set(range(len(PRIVATE_HANDS)))
    orbits: list[tuple[int, ...]] = []
    while unseen:
        seed = min(unseen)
        orbit = tuple(sorted({int(p[seed]) for p in HAND_PERMUTATIONS}))
        unseen.difference_update(orbit)
        orbits.append(orbit)
    return tuple(orbits)


def _fixed_subset_count(permutation: SuitPermutation, size: int) -> int:
    """Count size-card sets fixed by a suit permutation via cycle lengths."""
    seen: set[int] = set()
    suit_cycles: list[int] = []
    for suit in range(4):
        if suit in seen:
            continue
        current = suit
        length = 0
        while current not in seen:
            seen.add(current)
            length += 1
            current = permutation[current]
        suit_cycles.append(length)

    cycles = suit_cycles * len(RANKS)
    coefficients = [0] * (size + 1)
    coefficients[0] = 1
    for length in cycles:
        for degree in range(size, length - 1, -1):
            coefficients[degree] += coefficients[degree - length]
    return coefficients[size]


def board_orbit_count(card_count: int = 5) -> int:
    """Count suit-isomorphism classes using Burnside's lemma."""
    if not isinstance(card_count, int) or isinstance(card_count, bool) or not 0 <= card_count <= 52:
        raise ValueError("card_count must be an integer from 0 through 52")
    fixed_total = sum(_fixed_subset_count(g, card_count) for g in SUIT_PERMUTATIONS)
    return fixed_total // len(SUIT_PERMUTATIONS)
