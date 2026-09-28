"""Exact turn-to-river enumeration for heads-up Hold'em."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Iterator

import numpy as np

from cards import FULL_DECK, card_name, cards_in
from chance import ChanceState, canonical_chance_children
from river_kernel import river_kernel
from showdown import INCOMPATIBLE, LOSS, TIE, WIN, _validate_cards, compare_hands
from symmetry import SuitPermutation, canonical_card_sets, permute_cards


OUTCOME_NAMES = {LOSS: "loss", TIE: "tie", WIN: "win"}
SHOWDOWN_CACHE_SIZE = 262_144


@dataclass(frozen=True)
class RiverOutcome:
    """The result for one labelled river card, from the first hand's view."""

    river: int
    result: int

    @property
    def river_name(self) -> str:
        return card_name(self.river)

    @property
    def outcome(self) -> str:
        return OUTCOME_NAMES[self.result]


@dataclass(frozen=True)
class TurnResult:
    """All river outcomes and aggregate counts for a fixed heads-up deal."""

    turn: int
    first: int
    second: int
    rivers: tuple[RiverOutcome, ...]
    wins: int
    losses: int
    ties: int

    @property
    def total(self) -> int:
        return len(self.rivers)

    @property
    def equity(self) -> Fraction:
        """Return first-hand equity exactly, with ties split equally."""
        return Fraction(2 * self.wins + self.ties, 2 * self.total)


def river_children(turn: int, blocked: int = 0) -> tuple[tuple[int, int], ...]:
    """Return ``(river_card, completed_board)`` pairs in deck-bit order.

    ``blocked`` may contain known private cards.  With no blockers a turn has
    48 children; with two compatible private hands it has 44.
    """
    _validate_cards(turn, 4, "turn")
    if not isinstance(blocked, int) or isinstance(blocked, bool):
        raise TypeError("blocked must be an integer card bitset")
    if blocked < 0 or blocked & ~FULL_DECK:
        raise ValueError("blocked may only use bits 0 through 51")
    if turn & blocked:
        raise ValueError("blocked cards overlap the turn")

    unavailable = turn | blocked
    return tuple((river, turn | river) for river in cards_in(FULL_DECK ^ unavailable))


def turn_outcomes(turn: int, first: int, second: int) -> TurnResult | str:
    """Enumerate every river while evaluating suit-isomorphic deals once."""
    _validate_cards(turn, 4, "turn")
    _validate_cards(first, 2, "first")
    _validate_cards(second, 2, "second")
    if first & second:
        return INCOMPATIBLE
    if turn & first or turn & second:
        return INCOMPATIBLE

    rivers: list[RiverOutcome] = []
    state = ChanceState(turn, first, second)
    for group in canonical_chance_children(state):
        representative = group.representative.state
        result = _canonical_showdown(
            representative.first,
            representative.second,
            representative.board,
        )
        # All members remain present and labelled.  Symmetry only shares the
        # representative's calculation; it never removes a chance outcome.
        rivers.extend(
            RiverOutcome(member.child.reveal, result) for member in group.members
        )
    labelled_rivers = tuple(sorted(rivers, key=lambda item: item.river))
    wins = sum(item.result == WIN for item in labelled_rivers)
    losses = sum(item.result == LOSS for item in labelled_rivers)
    ties = sum(item.result == TIE for item in labelled_rivers)
    return TurnResult(turn, first, second, labelled_rivers, wins, losses, ties)


@lru_cache(maxsize=SHOWDOWN_CACHE_SIZE)
def _compare_canonical_deal(first: int, second: int, board: int) -> int:
    """Compare one already-canonical complete deal and cache its result."""
    return compare_hands(first, second, board)


def _canonical_showdown(first: int, second: int, board: int) -> int:
    canonical, _ = canonical_card_sets(first, second, board)
    return _compare_canonical_deal(canonical[0], canonical[1], canonical[2])


def showdown_cache_info():
    """Return cache statistics for canonical terminal comparisons."""
    return _compare_canonical_deal.cache_info()


def clear_showdown_cache() -> None:
    """Clear cached canonical terminal comparisons (mainly useful in tests)."""
    _compare_canonical_deal.cache_clear()


def permute_turn_result(
    result: TurnResult, permutation: SuitPermutation
) -> TurnResult:
    """Transport a complete labelled turn result through a suit permutation."""
    rivers = tuple(
        sorted(
            (
                RiverOutcome(permute_cards(item.river, permutation), item.result)
                for item in result.rivers
            ),
            key=lambda item: item.river,
        )
    )
    return TurnResult(
        permute_cards(result.turn, permutation),
        permute_cards(result.first, permutation),
        permute_cards(result.second, permutation),
        rivers,
        result.wins,
        result.losses,
        result.ties,
    )


def turn_river_kernels(
    turn: int,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield each river card and its existing compatibility/dominance kernel.

    Kernels are generated one at a time to avoid retaining all 48 matrices.
    Both matrix axes retain the fixed ordering in ``hand_space.PRIVATE_HANDS``.
    """
    for river, board in river_children(turn):
        compatibility, dominance = river_kernel(board)
        yield river, compatibility, dominance
