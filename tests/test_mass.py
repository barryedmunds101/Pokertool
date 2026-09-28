from fractions import Fraction

import pytest

from cards import make_hand, parse_card
from hand_space import HAND_INDEX, PRIVATE_HANDS
from mass import (
    JointEntry,
    JointMass,
    RangeMass,
    compatible_joint_mass,
    range_equity,
    range_is_invariant,
    transport_range,
    turn_mass,
    weighted_signature,
)
from river_kernel import river_kernel


def hand(*names: str) -> int:
    return make_hand(names)


TURN = hand("2c", "7d", "9h", "Js")
FIRST_STRONG = hand("Jc", "Ad")
FIRST_WEAK = hand("4d", "5d")
SECOND_STRONG = hand("9c", "Kd")
SECOND_WEAK = hand("8c", "Qd")
FIRST_RANGE = RangeMass.from_hands({FIRST_STRONG: 2, FIRST_WEAK: 1})
SECOND_RANGE = RangeMass.from_hands({SECOND_STRONG: 3, SECOND_WEAK: 1})


def test_range_mass_uses_fixed_1326_coordinates_and_exact_weights() -> None:
    assert len(FIRST_RANGE.weights) == 1326
    assert FIRST_RANGE.weights[HAND_INDEX[FIRST_STRONG]] == 2
    assert FIRST_RANGE.weights[HAND_INDEX[FIRST_WEAK]] == 1
    assert FIRST_RANGE.total == 3
    assert FIRST_RANGE.probability(FIRST_STRONG) == Fraction(2, 3)
    assert FIRST_RANGE.support == tuple(
        sorted((HAND_INDEX[FIRST_WEAK], HAND_INDEX[FIRST_STRONG]))
    )
    assert FIRST_RANGE.as_array().shape == (1326,)

    with pytest.raises(ValueError, match="non-negative"):
        RangeMass.from_hands({FIRST_STRONG: -1})


def test_product_joint_mass_is_sparse_exact_and_removes_blockers() -> None:
    first = RangeMass.from_hands(
        {FIRST_STRONG: 2, FIRST_WEAK: 1, hand("2c", "Ah"): 100}
    )
    joint = compatible_joint_mass(TURN, first, SECOND_RANGE)

    assert len(joint.entries) == 4
    assert joint.total == 12
    assert {entry.weight for entry in joint.entries} == {1, 2, 3, 6}
    assert joint.normalized().total == 1
    dense = joint.to_dense()
    assert dense.shape == (1326, 1326)
    assert dense[HAND_INDEX[FIRST_STRONG], HAND_INDEX[SECOND_STRONG]] == 6


def test_weighted_signature_matches_existing_river_operators() -> None:
    board = TURN | parse_card("6s")
    joint = compatible_joint_mass(board, FIRST_RANGE, SECOND_RANGE)
    signature = weighted_signature(board, joint)
    compatibility, dominance = river_kernel(board)

    denominator = Fraction()
    numerator = Fraction()
    for first_index in FIRST_RANGE.support:
        for second_index in SECOND_RANGE.support:
            weight = (
                FIRST_RANGE.weights[first_index] * SECOND_RANGE.weights[second_index]
            )
            denominator += weight * int(compatibility[first_index, second_index])
            numerator += weight * int(
                compatibility[first_index, second_index]
                + dominance[first_index, second_index]
            )

    assert signature.total == denominator
    assert signature.equity == Fraction(numerator, 2 * denominator)
    assert range_equity(board, FIRST_RANGE, SECOND_RANGE) == signature.equity


def test_turn_transport_conserves_mass_and_total_equity() -> None:
    result = turn_mass(TURN, FIRST_RANGE, SECOND_RANGE)

    assert len(result.rivers) == 48
    assert result.parent_mass.total == 12
    assert sum((river.probability for river in result.rivers), Fraction()) == 1
    assert (
        sum((river.joint_mass.total for river in result.rivers), Fraction())
        == result.parent_mass.total
    )
    assert result.average_river_equity == result.equity
    assert len({river.probability for river in result.rivers}) > 1
    assert all(
        river.conditional_beliefs is None
        or river.conditional_beliefs.total == 1
        for river in result.rivers
    )


def test_one_fixed_deal_has_four_impossible_and_44_equiprobable_rivers() -> None:
    first = RangeMass.uniform([FIRST_STRONG])
    second = RangeMass.uniform([SECOND_STRONG])
    result = turn_mass(TURN, first, second)

    probabilities = [river.probability for river in result.rivers]
    assert probabilities.count(0) == 4
    assert probabilities.count(Fraction(1, 44)) == 44


def test_ranges_may_break_suit_symmetry_and_can_be_transported() -> None:
    permutation = (1, 0, 2, 3)
    asymmetric = RangeMass.uniform([FIRST_STRONG])
    transported = transport_range(asymmetric, permutation)

    assert transported.total == asymmetric.total
    assert not range_is_invariant(asymmetric, permutation)
    assert range_is_invariant(RangeMass.uniform(PRIVATE_HANDS), permutation)
    assert transport_range(transported, permutation) == asymmetric


def test_arbitrary_correlated_joint_mass_is_representable() -> None:
    correlated = JointMass(
        (
            JointEntry(HAND_INDEX[FIRST_STRONG], HAND_INDEX[SECOND_WEAK], Fraction(3, 4)),
            JointEntry(HAND_INDEX[FIRST_WEAK], HAND_INDEX[SECOND_STRONG], Fraction(1, 4)),
        )
    )

    assert correlated.total == 1
    assert len(correlated.entries) == 2
