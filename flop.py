"""Exact flop-to-river enumeration, defined through turn outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from cards import FULL_DECK, card_name, cards_in
from showdown import INCOMPATIBLE, _validate_cards
from turn import TurnResult, turn_outcomes


@dataclass(frozen=True)
class TurnOutcome:
    """One labelled turn card and all of its river outcomes."""

    turn_card: int
    result: TurnResult

    @property
    def turn_name(self) -> str:
        return card_name(self.turn_card)


@dataclass(frozen=True)
class FlopResult:
    """The recursively aggregated result for a fixed flop and two hands."""

    flop: int
    first: int
    second: int
    turns: tuple[TurnOutcome, ...]
    wins: int
    losses: int
    ties: int

    @property
    def total(self) -> int:
        return self.wins + self.losses + self.ties

    @property
    def equity(self) -> Fraction:
        return Fraction(2 * self.wins + self.ties, 2 * self.total)


def turn_children(flop: int, blocked: int = 0) -> tuple[tuple[int, int], ...]:
    """Return ``(turn_card, four_card_board)`` children in deck-bit order."""
    _validate_cards(flop, 3, "flop")
    if not isinstance(blocked, int) or isinstance(blocked, bool):
        raise TypeError("blocked must be an integer card bitset")
    if blocked < 0 or blocked & ~FULL_DECK:
        raise ValueError("blocked may only use bits 0 through 51")
    if flop & blocked:
        raise ValueError("blocked cards overlap the flop")

    unavailable = flop | blocked
    return tuple((card, flop | card) for card in cards_in(FULL_DECK ^ unavailable))


def flop_outcomes(flop: int, first: int, second: int) -> FlopResult | str:
    """Aggregate every turn recursively through :func:`turn_outcomes`.

    There are 45 labelled turn children and 44 rivers below each child, so a
    compatible deal contains 1,980 ordered turn-river runouts.
    """
    _validate_cards(flop, 3, "flop")
    _validate_cards(first, 2, "first")
    _validate_cards(second, 2, "second")
    if first & second or flop & (first | second):
        return INCOMPATIBLE

    turns: list[TurnOutcome] = []
    for turn_card, turn_board in turn_children(flop, first | second):
        result = turn_outcomes(turn_board, first, second)
        if result == INCOMPATIBLE:  # Protected by the compatibility check.
            raise RuntimeError("compatible flop produced an incompatible turn")
        turns.append(TurnOutcome(turn_card, result))

    children = tuple(turns)
    return FlopResult(
        flop,
        first,
        second,
        children,
        sum(child.result.wins for child in children),
        sum(child.result.losses for child in children),
        sum(child.result.ties for child in children),
    )
