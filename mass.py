"""Exact private-range, joint mass and chance propagation on the hand space."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Iterator, Mapping

import numpy as np

from cards import card_name
from chance import ChanceState, chance_child_count, chance_children
from compatibility import _validate_card_set
from hand_space import HAND_INDEX, PRIVATE_HANDS
from showdown import LOSS, TIE, WIN, _validate_cards, compare_hands
from symmetry import SuitPermutation, hand_permutation


Exact = int | Fraction


@dataclass(frozen=True)
class RangeMass:
    """An unnormalised exact weight on each of the 1,326 private hands."""

    weights: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        if len(self.weights) != len(PRIVATE_HANDS):
            raise ValueError(f"range must have {len(PRIVATE_HANDS)} coordinates")
        converted = tuple(Fraction(weight) for weight in self.weights)
        if any(weight < 0 for weight in converted):
            raise ValueError("range weights must be non-negative")
        object.__setattr__(self, "weights", converted)

    @classmethod
    def from_hands(cls, weights: Mapping[int, Exact]) -> "RangeMass":
        """Build a range from private-hand bitsets and exact weights."""
        vector = [Fraction()] * len(PRIVATE_HANDS)
        for hand, weight in weights.items():
            try:
                index = HAND_INDEX[hand]
            except (KeyError, TypeError):
                raise ValueError("range keys must be two-card private hands") from None
            exact = Fraction(weight)
            if exact < 0:
                raise ValueError("range weights must be non-negative")
            vector[index] = exact
        return cls(tuple(vector))

    @classmethod
    def uniform(cls, hands: Iterable[int]) -> "RangeMass":
        """Assign unit mass to each supplied private hand."""
        return cls.from_hands({hand: 1 for hand in hands})

    @property
    def total(self) -> Fraction:
        return sum(self.weights, Fraction())

    @property
    def support(self) -> tuple[int, ...]:
        """Return nonzero ambient hand indices."""
        return tuple(index for index, weight in enumerate(self.weights) if weight)

    def probability(self, hand: int) -> Fraction:
        """Return the normalized marginal probability of one hand."""
        if not self.total:
            raise ValueError("an empty range has no probability distribution")
        try:
            index = HAND_INDEX[hand]
        except (KeyError, TypeError):
            raise ValueError("hand must be a two-card private hand") from None
        return self.weights[index] / self.total

    def as_array(self) -> np.ndarray:
        """Return the fixed-coordinate vector as an exact object array."""
        return np.asarray(self.weights, dtype=object)


@dataclass(frozen=True)
class JointEntry:
    first_index: int
    second_index: int
    weight: Fraction


@dataclass(frozen=True)
class JointMass:
    """Sparse exact mass on ordered pairs of private-hand coordinates."""

    entries: tuple[JointEntry, ...]

    def __post_init__(self) -> None:
        seen: set[tuple[int, int]] = set()
        converted: list[JointEntry] = []
        for entry in self.entries:
            if not 0 <= entry.first_index < len(PRIVATE_HANDS):
                raise ValueError("first hand index is outside the ambient space")
            if not 0 <= entry.second_index < len(PRIVATE_HANDS):
                raise ValueError("second hand index is outside the ambient space")
            key = (entry.first_index, entry.second_index)
            if key in seen:
                raise ValueError("joint coordinates must be unique")
            seen.add(key)
            weight = Fraction(entry.weight)
            if weight < 0:
                raise ValueError("joint weights must be non-negative")
            if weight:
                converted.append(JointEntry(*key, weight))
        object.__setattr__(self, "entries", tuple(converted))

    @property
    def total(self) -> Fraction:
        return sum((entry.weight for entry in self.entries), Fraction())

    def normalized(self) -> "JointMass":
        """Return conditional beliefs with total mass one."""
        total = self.total
        if not total:
            raise ValueError("zero joint mass cannot be normalized")
        return JointMass(
            tuple(
                JointEntry(entry.first_index, entry.second_index, entry.weight / total)
                for entry in self.entries
            )
        )

    def scaled(self, factor: Exact) -> "JointMass":
        exact = Fraction(factor)
        if exact < 0:
            raise ValueError("joint mass cannot be scaled by a negative factor")
        return JointMass(
            tuple(
                JointEntry(entry.first_index, entry.second_index, entry.weight * exact)
                for entry in self.entries
            )
        )

    def to_dense(self) -> np.ndarray:
        """Materialize the 1,326 × 1,326 exact matrix when explicitly needed."""
        matrix = np.full((len(PRIVATE_HANDS), len(PRIVATE_HANDS)), Fraction(), dtype=object)
        for entry in self.entries:
            matrix[entry.first_index, entry.second_index] = entry.weight
        return matrix


@dataclass(frozen=True)
class JointMassChild:
    """One labelled chance child carrying its unnormalised joint mass."""

    reveal: int
    state: ChanceState
    joint_mass: JointMass


@dataclass(frozen=True)
class WeightedSignature:
    wins: Fraction
    losses: Fraction
    ties: Fraction

    @property
    def total(self) -> Fraction:
        return self.wins + self.losses + self.ties

    @property
    def equity(self) -> Fraction:
        if not self.total:
            raise ValueError("zero mass has no equity")
        return Fraction(2 * self.wins + self.ties, 2 * self.total)


@dataclass(frozen=True)
class RiverMass:
    """One river's probability, distributed joint mass and weighted result."""

    river: int
    probability: Fraction
    joint_mass: JointMass
    signature: WeightedSignature

    @property
    def river_name(self) -> str:
        return card_name(self.river)

    @property
    def conditional_beliefs(self) -> JointMass | None:
        """Return normalized pair beliefs, or ``None`` for an impossible river."""
        return self.joint_mass.normalized() if self.joint_mass.total else None


@dataclass(frozen=True)
class TurnMass:
    turn: int
    parent_mass: JointMass
    rivers: tuple[RiverMass, ...]

    @property
    def signature(self) -> WeightedSignature:
        return WeightedSignature(
            sum((river.signature.wins for river in self.rivers), Fraction()),
            sum((river.signature.losses for river in self.rivers), Fraction()),
            sum((river.signature.ties for river in self.rivers), Fraction()),
        )

    @property
    def equity(self) -> Fraction:
        return self.signature.equity

    @property
    def average_river_equity(self) -> Fraction:
        return sum(
            (
                river.probability * river.signature.equity
                for river in self.rivers
                if river.probability
            ),
            Fraction(),
        )


def compatible_joint_mass(board: int, first: RangeMass, second: RangeMass) -> JointMass:
    """Return ``M_B(i,j) = x_i C_B(i,j) y_j`` as sparse exact mass."""
    _validate_card_set(board, "board")
    entries: list[JointEntry] = []
    for first_index in first.support:
        first_hand = PRIVATE_HANDS[first_index]
        if first_hand & board:
            continue
        for second_index in second.support:
            second_hand = PRIVATE_HANDS[second_index]
            if second_hand & board or first_hand & second_hand:
                continue
            entries.append(
                JointEntry(
                    first_index,
                    second_index,
                    first.weights[first_index] * second.weights[second_index],
                )
            )
    return JointMass(tuple(entries))


def propagate_joint_mass(
    state: ChanceState, joint_mass: JointMass
) -> Iterator[JointMassChild]:
    """Distribute arbitrary exact joint mass over one street of chance children.

    Each compatible private-hand pair splits equally across its legal reveals:
    17,296 flops preflop, 45 turns on a flop, or 44 rivers on a turn. All labels
    from ``chance_children(state)`` are retained, including zero-mass children.
    The state's unconditioned child probabilities must not be used to split
    a pair's mass: the pair supplies both players' blockers.

    For every nonterminal input, summing children recovers the parent weight
    at every hand-pair coordinate. No normalization, showdown evaluation or
    recursive expansion is performed. Child masses are yielded one at a time;
    dense input supports can still be expensive. Empty mass is supported, and
    a river yields no children (its mass remains terminal).

    On iteration, reject positive-mass deals incompatible with the board or
    fixed hands in ``state`` rather than silently discarding their mass.
    """
    deals: list[tuple[JointEntry, int]] = []
    for entry in joint_mass.entries:
        first = PRIVATE_HANDS[entry.first_index]
        second = PRIVATE_HANDS[entry.second_index]
        if first & second or (first | second) & state.board:
            raise ValueError("joint mass contains a deal incompatible with the board")
        if (state.first and first != state.first) or (
            state.second and second != state.second
        ):
            raise ValueError("joint mass does not match the state's fixed hands")
        deals.append((entry, first | second))

    if state.terminal:
        return

    # Every legal pair contains exactly four private cards. One representative
    # supplies the shared conditional count through the existing chance layer.
    if deals:
        entry = deals[0][0]
        pair_state = ChanceState(
            state.board,
            PRIVATE_HANDS[entry.first_index],
            PRIVATE_HANDS[entry.second_index],
            state.reveals,
            state.action_history,
        )
        probability = Fraction(1, chance_child_count(pair_state))
        shares = tuple(
            (
                JointEntry(item.first_index, item.second_index, item.weight * probability),
                blocked,
            )
            for item, blocked in deals
        )
    else:
        shares = ()

    for child in chance_children(state):
        mass = JointMass(
            tuple(entry for entry, blocked in shares if not blocked & child.reveal)
        )
        yield JointMassChild(child.reveal, child.state, mass)


def weighted_signature(board: int, joint_mass: JointMass) -> WeightedSignature:
    """Calculate exact weighted W/L/T on a complete river board."""
    _validate_cards(board, 5, "board")
    wins = losses = ties = Fraction()
    for entry in joint_mass.entries:
        first = PRIVATE_HANDS[entry.first_index]
        second = PRIVATE_HANDS[entry.second_index]
        if first & board or second & board or first & second:
            raise ValueError("joint mass contains a deal incompatible with the board")
        result = compare_hands(first, second, board)
        if result == WIN:
            wins += entry.weight
        elif result == LOSS:
            losses += entry.weight
        else:
            ties += entry.weight
    return WeightedSignature(wins, losses, ties)


def range_equity(board: int, first: RangeMass, second: RangeMass) -> Fraction:
    """Return ``xᵀ(C+D)y / (2 xᵀCy)`` exactly for a river board."""
    joint = compatible_joint_mass(board, first, second)
    return weighted_signature(board, joint).equity


def turn_mass(turn: int, first: RangeMass, second: RangeMass) -> TurnMass:
    """Distribute compatible turn mass across all 48 public river cards.

    A fixed compatible hand pair has 44 legal rivers, so each pair's mass is
    divided by 44. River probabilities emerge after summing those contributions
    and therefore reflect blockers in the weighted ranges.
    """
    _validate_cards(turn, 4, "turn")
    parent = compatible_joint_mass(turn, first, second)
    parent_total = parent.total
    if not parent_total:
        raise ValueError("the ranges have no compatible mass on this turn")

    rivers: list[RiverMass] = []
    for child in propagate_joint_mass(ChanceState(turn), parent):
        child_mass = child.joint_mass
        probability = child_mass.total / parent_total
        signature = weighted_signature(child.state.board, child_mass)
        rivers.append(RiverMass(child.reveal, probability, child_mass, signature))
    return TurnMass(turn, parent, tuple(rivers))


def transport_range(range_mass: RangeMass, permutation: SuitPermutation) -> RangeMass:
    """Transport range weights through the ambient hand permutation."""
    indices = hand_permutation(permutation)
    weights = [Fraction()] * len(PRIVATE_HANDS)
    for old_index, new_index in enumerate(indices):
        weights[int(new_index)] = range_mass.weights[old_index]
    return RangeMass(tuple(weights))


def range_is_invariant(range_mass: RangeMass, permutation: SuitPermutation) -> bool:
    """Return whether the range respects a given suit symmetry."""
    return transport_range(range_mass, permutation) == range_mass
