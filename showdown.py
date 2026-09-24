"""Exact Texas Hold'em showdown comparison."""

from __future__ import annotations

from collections import Counter
from itertools import combinations

from cards import FULL_DECK, cards_in
from compatibility import compatible


LOSS = 0
TIE = 1
WIN = 2
INCOMPATIBLE = "I"

# Every unordered two-card hand, in stable deck-bit order.  The same ordering
# is used for comparison vectors and for both axes of comparison matrices.
PRIVATE_HANDS = tuple(first | second for first, second in combinations(cards_in(FULL_DECK), 2))


def _validate_cards(value: int, expected_count: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer card bitset")
    if value < 0 or value & ~FULL_DECK:
        raise ValueError(f"{name} may only use bits 0 through 51")
    if value.bit_count() != expected_count:
        raise ValueError(
            f"{name} must contain exactly {expected_count} cards; "
            f"got {value.bit_count()}"
        )


def _five_card_rank(hand: int) -> tuple[int, ...]:
    """Return a lexicographically comparable rank for exactly five cards."""
    card_indexes = [value.bit_length() - 1 for value in cards_in(hand)]
    ranks = [index % 13 for index in card_indexes]
    suits = [index // 13 for index in card_indexes]
    counts = Counter(ranks)

    # Groups compare count first, then rank: e.g. pairs before kickers.
    groups = sorted(((count, rank) for rank, count in counts.items()), reverse=True)
    is_flush = len(set(suits)) == 1

    unique_ranks = set(ranks)
    if {12, 0, 1, 2, 3}.issubset(unique_ranks):
        straight_high = 3  # Five-high straight: A-2-3-4-5.
    elif len(unique_ranks) == 5 and max(unique_ranks) - min(unique_ranks) == 4:
        straight_high = max(unique_ranks)
    else:
        straight_high = None

    if is_flush and straight_high is not None:
        return (8, straight_high)
    if groups[0][0] == 4:
        return (7, groups[0][1], groups[1][1])
    if groups[0][0] == 3 and groups[1][0] == 2:
        return (6, groups[0][1], groups[1][1])
    if is_flush:
        return (5, *sorted(ranks, reverse=True))
    if straight_high is not None:
        return (4, straight_high)
    if groups[0][0] == 3:
        kickers = sorted((rank for rank in ranks if rank != groups[0][1]), reverse=True)
        return (3, groups[0][1], *kickers)
    if groups[0][0] == 2 and groups[1][0] == 2:
        high_pair = max(groups[0][1], groups[1][1])
        low_pair = min(groups[0][1], groups[1][1])
        kicker = next(rank for rank, count in counts.items() if count == 1)
        return (2, high_pair, low_pair, kicker)
    if groups[0][0] == 2:
        pair = groups[0][1]
        kickers = sorted((rank for rank in ranks if rank != pair), reverse=True)
        return (1, pair, *kickers)
    return (0, *sorted(ranks, reverse=True))


def best_hand_rank(cards: int) -> tuple[int, ...]:
    """Return the best five-card rank available from five to seven cards."""
    if not isinstance(cards, int) or isinstance(cards, bool):
        raise TypeError("cards must be an integer card bitset")
    if cards < 0 or cards & ~FULL_DECK:
        raise ValueError("cards may only use bits 0 through 51")
    if not 5 <= cards.bit_count() <= 7:
        raise ValueError("cards must contain between five and seven cards")

    return max(
        _five_card_rank(sum(combo))
        for combo in combinations(cards_in(cards), 5)
    )


def compare_hands(first: int, second: int, board: int) -> int:
    """Compare two private hands on a complete five-card river board.

    The result is from ``first``'s perspective: ``LOSS`` (0), ``TIE`` (1), or
    ``WIN`` (2).  All three card sets must be mutually compatible.
    """
    _validate_cards(first, 2, "first")
    _validate_cards(second, 2, "second")
    _validate_cards(board, 5, "board")

    if not compatible(first, second):
        raise ValueError("The private hands overlap")
    if not compatible(first, board) or not compatible(second, board):
        raise ValueError("A private hand overlaps the board")

    first_rank = best_hand_rank(first | board)
    second_rank = best_hand_rank(second | board)
    if first_rank < second_rank:
        return LOSS
    if first_rank == second_rank:
        return TIE
    return WIN


def _ranks_for_board(board: int) -> dict[int, tuple[int, ...]]:
    """Precompute the showdown rank of every private hand valid on ``board``."""
    return {
        hand: best_hand_rank(hand | board)
        for hand in PRIVATE_HANDS
        if not hand & board
    }


def comparison_vector(first: int, board: int) -> list[int | str]:
    """Compare ``first`` with every possible private hand on ``board``.

    Entries follow :data:`PRIVATE_HANDS`.  A result is ``LOSS`` (0), ``TIE``
    (1), or ``WIN`` (2) from ``first``'s perspective.  ``INCOMPATIBLE``
    (``"I"``) is used when a private hand overlaps ``first`` or the board.
    """
    _validate_cards(first, 2, "first")
    _validate_cards(board, 5, "board")

    if first & board:
        return [INCOMPATIBLE] * len(PRIVATE_HANDS)

    ranks = _ranks_for_board(board)
    first_rank = ranks[first]
    result: list[int | str] = []
    for second in PRIVATE_HANDS:
        if second not in ranks or first & second:
            result.append(INCOMPATIBLE)
        else:
            second_rank = ranks[second]
            result.append(LOSS if first_rank < second_rank else TIE if first_rank == second_rank else WIN)
    return result


def comparison_matrix(board: int) -> list[list[int | str]]:
    """Return all private-hand comparisons for a complete river ``board``.

    Rows and columns both follow :data:`PRIVATE_HANDS`; cell ``[row][column]``
    is from the row hand's perspective.  Any overlap with the board or between
    the two hands produces ``INCOMPATIBLE`` (``"I"``).
    """
    _validate_cards(board, 5, "board")
    ranks = _ranks_for_board(board)
    matrix: list[list[int | str]] = []

    for first in PRIVATE_HANDS:
        if first not in ranks:
            matrix.append([INCOMPATIBLE] * len(PRIVATE_HANDS))
            continue

        first_rank = ranks[first]
        row: list[int | str] = []
        for second in PRIVATE_HANDS:
            if second not in ranks or first & second:
                row.append(INCOMPATIBLE)
            else:
                second_rank = ranks[second]
                row.append(LOSS if first_rank < second_rank else TIE if first_rank == second_rank else WIN)
        matrix.append(row)

    return matrix


# Backwards-compatible name retained for existing callers.
compare_private_hands = compare_hands
