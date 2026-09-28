"""Non-recursive, exact chance transitions for Texas Hold'em streets."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from math import comb
from typing import TypeAlias

from cards import FULL_DECK, cards_in
from compatibility import _validate_card_set
from showdown import _validate_cards
from symmetry import (
    SUIT_PERMUTATIONS,
    SuitPermutation,
    canonical_card_sets,
    permute_cards,
)


IDENTITY: SuitPermutation = (0, 1, 2, 3)
STREETS = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}
CanonicalKey: TypeAlias = tuple[int, int, int, tuple[int, ...], tuple[str, ...]]


@dataclass(frozen=True)
class ChanceState:
    """A public board plus optional fixed hands and preserved history.

    ``reveals`` records the street-by-street public observations: an unordered
    three-card flop followed by one-card turn and river reveals.  The opaque
    ``action_history`` is carried into every child so future game-tree layers
    can distinguish identical boards reached after different betting actions.
    """

    board: int = 0
    first: int = 0
    second: int = 0
    reveals: tuple[int, ...] = ()
    action_history: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_card_set(self.board, "board")
        if self.board.bit_count() not in STREETS:
            raise ValueError("board must contain 0, 3, 4, or 5 cards")
        for hand, name in ((self.first, "first"), (self.second, "second")):
            _validate_card_set(hand, name)
            if hand and hand.bit_count() != 2:
                raise ValueError(f"{name} must be empty or contain exactly 2 cards")
        if self.first & self.second:
            raise ValueError("the private hands overlap")
        if self.board & (self.first | self.second):
            raise ValueError("a private hand overlaps the board")
        if self.reveals:
            expected = {0: (), 3: (3,), 4: (3, 1), 5: (3, 1, 1)}[
                self.board.bit_count()
            ]
            if tuple(reveal.bit_count() for reveal in self.reveals) != expected:
                raise ValueError("reveals do not match the board's street history")
            combined = 0
            for reveal in self.reveals:
                if combined & reveal:
                    raise ValueError("reveals must be disjoint")
                combined |= reveal
            if combined != self.board:
                raise ValueError("reveals must be disjoint and combine to the board")

    @property
    def street(self) -> str:
        return STREETS[self.board.bit_count()]

    @property
    def blocked(self) -> int:
        return self.first | self.second

    @property
    def terminal(self) -> bool:
        return self.board.bit_count() == 5


@dataclass(frozen=True)
class ChanceChild:
    """One legal reveal and its exact conditional probability."""

    reveal: int
    state: ChanceState
    probability: Fraction

    @property
    def canonical_key(self) -> CanonicalKey:
        return canonical_state(self.state)[0]

    @property
    def canonical_transport(self) -> SuitPermutation:
        return canonical_state(self.state)[1]


@dataclass(frozen=True)
class ChildTransport:
    """A raw child and a parent-stabilizer map to its representative."""

    child: ChanceChild
    permutation: SuitPermutation


@dataclass(frozen=True)
class CanonicalChanceChild:
    """One conditional-symmetry orbit of legal chance children."""

    representative: ChanceChild
    probability: Fraction
    multiplicity: int
    members: tuple[ChildTransport, ...]


def _state_parts(state: ChanceState) -> tuple[int, ...]:
    return (state.board, state.first, state.second, *state.reveals)


def _state_key(state: ChanceState) -> CanonicalKey:
    return (
        state.board,
        state.first,
        state.second,
        state.reveals,
        state.action_history,
    )


def permute_state(state: ChanceState, permutation: SuitPermutation) -> ChanceState:
    """Transport every card-bearing component while preserving actions."""
    return ChanceState(
        permute_cards(state.board, permutation),
        permute_cards(state.first, permutation),
        permute_cards(state.second, permutation),
        tuple(permute_cards(reveal, permutation) for reveal in state.reveals),
        state.action_history,
    )


def canonical_state(state: ChanceState) -> tuple[CanonicalKey, SuitPermutation]:
    """Return a stable global key and the permutation transporting to it."""
    _, permutation = canonical_card_sets(*_state_parts(state))
    moved = permute_state(state, permutation)
    return _state_key(moved), permutation


def state_stabilizer(state: ChanceState) -> tuple[SuitPermutation, ...]:
    """Return permutations preserving this conditioned state and its history."""
    return tuple(
        permutation
        for permutation in SUIT_PERMUTATIONS
        if permute_state(state, permutation) == state
    )


def chance_children(state: ChanceState) -> tuple[ChanceChild, ...]:
    """Return only the next legal chance states, never their descendants."""
    if state.terminal:
        return ()

    available = cards_in(FULL_DECK ^ (state.board | state.blocked))
    reveal_count = 3 if state.street == "preflop" else 1
    reveals = (
        (combo[0] | combo[1] | combo[2] for combo in combinations(available, 3))
        if reveal_count == 3
        else iter(available)
    )
    child_count = comb(len(available), reveal_count)
    probability = Fraction(1, child_count)
    children: list[ChanceChild] = []
    for reveal in reveals:
        if state.street == "preflop":
            history = (reveal,)
        elif state.reveals:
            history = state.reveals + (reveal,)
        else:
            # A state constructed from a bare board has no recoverable reveal
            # order, so its children remain board-only states too.
            history = ()
        child_state = ChanceState(
            state.board | reveal,
            state.first,
            state.second,
            history,
            state.action_history,
        )
        children.append(ChanceChild(reveal, child_state, probability))
    return tuple(children)


def canonical_chance_children(state: ChanceState) -> tuple[CanonicalChanceChild, ...]:
    """Group next children under the current state's stabilizer.

    Grouping under the stabilizer, rather than the full suit group, preserves
    the board and any fixed blockers.  Each member stores a permutation that
    maps it to the selected representative.
    """
    raw = chance_children(state)
    if not raw:
        return ()
    stabilizer = state_stabilizer(state)
    by_reveal = {child.reveal: child for child in raw}
    unseen = set(by_reveal)
    groups: list[CanonicalChanceChild] = []
    while unseen:
        seed = min(unseen)
        orbit = {permute_cards(seed, permutation) for permutation in stabilizer}
        member_reveals = tuple(sorted(orbit & unseen))
        representative_reveal = min(orbit)
        representative = by_reveal[representative_reveal]
        transports: list[ChildTransport] = []
        for reveal in member_reveals:
            child = by_reveal[reveal]
            transport = next(
                permutation
                for permutation in stabilizer
                if permute_cards(reveal, permutation) == representative_reveal
            )
            transports.append(ChildTransport(child, transport))
        unseen.difference_update(member_reveals)
        multiplicity = len(member_reveals)
        groups.append(
            CanonicalChanceChild(
                representative,
                raw[0].probability * multiplicity,
                multiplicity,
                tuple(transports),
            )
        )
    return tuple(groups)
