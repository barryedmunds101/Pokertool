"""Exact preflop enumeration, recursively aggregated through flop outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Iterator

from cards import FULL_DECK, cards_in, hand_names
from flop import FlopResult, flop_outcomes
from showdown import INCOMPATIBLE, _validate_cards


@dataclass(frozen=True)
class FlopOutcome:
    """A labelled flop and its aggregate turn-to-river counts."""

    flop: int
    wins: int
    losses: int
    ties: int

    @classmethod
    def from_result(cls, result: FlopResult) -> "FlopOutcome":
        return cls(result.flop, result.wins, result.losses, result.ties)

    @property
    def flop_name(self) -> str:
        return " ".join(hand_names(self.flop))

    @property
    def total(self) -> int:
        return self.wins + self.losses + self.ties

    @property
    def equity(self) -> Fraction:
        return Fraction(2 * self.wins + self.ties, 2 * self.total)


@dataclass(frozen=True)
class PreflopResult:
    """Exact result over all labelled flop, turn and river runouts."""

    first: int
    second: int
    flops: tuple[FlopOutcome, ...]
    wins: int
    losses: int
    ties: int

    @property
    def total(self) -> int:
        return self.wins + self.losses + self.ties

    @property
    def equity(self) -> Fraction:
        return Fraction(2 * self.wins + self.ties, 2 * self.total)


def flop_children(first: int, second: int) -> Iterator[int]:
    """Yield all 17,296 unordered flops compatible with two fixed hands."""
    _validate_cards(first, 2, "first")
    _validate_cards(second, 2, "second")
    if first & second:
        return
    available = cards_in(FULL_DECK ^ (first | second))
    for flop_cards in combinations(available, 3):
        yield flop_cards[0] | flop_cards[1] | flop_cards[2]


def preflop_flop_outcomes(first: int, second: int) -> Iterator[FlopOutcome]:
    """Yield labelled flop summaries lazily using :func:`flop_outcomes`.

    This is the practical interface for inspecting or checkpointing the large
    preflop tree without retaining every turn and river record in memory.
    """
    _validate_cards(first, 2, "first")
    _validate_cards(second, 2, "second")
    if first & second:
        return
    for flop in flop_children(first, second):
        result = flop_outcomes(flop, first, second)
        if result == INCOMPATIBLE:  # Protected by flop_children.
            raise RuntimeError("compatible preflop deal produced an incompatible flop")
        yield FlopOutcome.from_result(result)


def preflop_outcomes(first: int, second: int) -> PreflopResult | str:
    """Recursively aggregate the complete exact preflop chance tree.

    The calculation visits 17,296 flops and 34,246,080 ordered turn-river
    runouts.  Use :func:`preflop_flop_outcomes` when incremental consumption is
    preferable to waiting for and retaining the complete result.
    """
    _validate_cards(first, 2, "first")
    _validate_cards(second, 2, "second")
    if first & second:
        return INCOMPATIBLE

    flops = tuple(preflop_flop_outcomes(first, second))
    return PreflopResult(
        first,
        second,
        flops,
        sum(flop.wins for flop in flops),
        sum(flop.losses for flop in flops),
        sum(flop.ties for flop in flops),
    )
