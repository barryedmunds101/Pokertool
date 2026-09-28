"""Exact preflop enumeration, recursively aggregated through flop outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterator

from cards import hand_names
from chance import ChanceState, canonical_chance_children, chance_children
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
    yield from (
        child.reveal for child in chance_children(ChanceState(first=first, second=second))
    )


def preflop_flop_outcomes(first: int, second: int) -> Iterator[FlopOutcome]:
    """Yield labelled flop summaries lazily using :func:`flop_outcomes`.

    This is the practical interface for inspecting or checkpointing the large
    preflop tree without retaining every turn and river record in memory.
    """
    _validate_cards(first, 2, "first")
    _validate_cards(second, 2, "second")
    if first & second:
        return
    state = ChanceState(first=first, second=second)
    groups = canonical_chance_children(state)
    group_for_reveal = {
        member.child.reveal: group
        for group in groups
        for member in group.members
    }
    summaries: dict[int, FlopOutcome] = {}
    for child in chance_children(state):
        group = group_for_reveal[child.reveal]
        representative_flop = group.representative.reveal
        if representative_flop not in summaries:
            result = flop_outcomes(representative_flop, first, second)
            if result == INCOMPATIBLE:  # Protected by chance_children.
                raise RuntimeError("compatible preflop deal produced an incompatible flop")
            summaries[representative_flop] = FlopOutcome.from_result(result)
        summary = summaries[representative_flop]
        yield FlopOutcome(child.reveal, summary.wins, summary.losses, summary.ties)


def preflop_outcomes(first: int, second: int) -> PreflopResult | str:
    """Recursively aggregate the complete exact preflop chance tree.

    The result represents all 17,296 flops and 34,246,080 ordered turn-river
    runouts. Suit-isomorphic children share calculations under each conditioned
    state's stabilizer. Use :func:`preflop_flop_outcomes` when incremental
    consumption is preferable to waiting for and retaining the complete result.
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
