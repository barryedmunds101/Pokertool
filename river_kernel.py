"""Numeric exact-river chance operators on the ambient private-hand space."""

from __future__ import annotations

import numpy as np

from cards import FULL_DECK
from hand_space import PRIVATE_HANDS
from showdown import _ranks_for_board, _validate_cards


def river_kernel(board: int) -> tuple[np.ndarray, np.ndarray]:
    """Return compatibility ``C`` and dominance ``D`` for a river board.

    Both axes use the fixed ordering in :data:`hand_space.PRIVATE_HANDS`.
    ``C[i, j]`` is true exactly when both hands and the board can coexist.
    On compatible cells, ``D`` is -1, 0, or +1 according as the row hand
    loses, ties, or wins; incompatible cells are zero.
    """
    _validate_cards(board, 5, "board")
    hands = np.fromiter(PRIVATE_HANDS, dtype=np.uint64)
    board_value = np.uint64(board & FULL_DECK)
    board_valid = np.bitwise_and(hands, board_value) == 0
    disjoint = np.bitwise_and(hands[:, None], hands[None, :]) == 0
    compatibility = disjoint & board_valid[:, None] & board_valid[None, :]

    ranks = _ranks_for_board(board)
    ordered_ranks = {rank: value for value, rank in enumerate(sorted(set(ranks.values())))}
    strengths = np.full(len(PRIVATE_HANDS), -1, dtype=np.int16)
    for index, hand in enumerate(PRIVATE_HANDS):
        if hand in ranks:
            strengths[index] = ordered_ranks[ranks[hand]]

    dominance = np.sign(strengths[:, None] - strengths[None, :]).astype(np.int8)
    dominance[~compatibility] = 0
    return compatibility, dominance


def result_matrix(board: int) -> np.ndarray:
    """Return numeric doubled-equity ``R = C + D`` with invalid cells zero."""
    compatibility, dominance = river_kernel(board)
    return compatibility.astype(np.int8) + dominance
