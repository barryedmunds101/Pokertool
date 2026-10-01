from collections import defaultdict
from fractions import Fraction

import pytest

import mass as mass_module
from cards import make_hand, parse_card
from chance import ChanceState, chance_child_count, chance_children, permute_state
from hand_space import HAND_INDEX, PRIVATE_HANDS
from mass import JointEntry, JointMass, propagate_joint_mass
from symmetry import permute_cards


FLOP = make_hand(["2c", "7d", "9h"])
TURN = FLOP | parse_card("Js")
FIRST = make_hand(["Jc", "Ad"])
SECOND = make_hand(["9c", "Kd"])
OTHER_FIRST = make_hand(["4d", "5d"])
OTHER_SECOND = make_hand(["8c", "Qd"])


def pair_mass(first=FIRST, second=SECOND, weight=Fraction(5, 3)):
    return JointMass((JointEntry(HAND_INDEX[first], HAND_INDEX[second], weight),))


def coordinates(joint_mass):
    return {
        (entry.first_index, entry.second_index): entry.weight
        for entry in joint_mass.entries
    }


def add_masses(*masses):
    weights = defaultdict(Fraction)
    for joint_mass in masses:
        for key, weight in coordinates(joint_mass).items():
            weights[key] += weight
    return JointMass(tuple(JointEntry(*key, weight) for key, weight in weights.items()))


CORRELATED = add_masses(pair_mass(), pair_mass(OTHER_FIRST, OTHER_SECOND, Fraction(7, 5)))


@pytest.mark.parametrize(
    "board,label_count,positive_count",
    [(0, 22100, 17296), (FLOP, 49, 45), (TURN, 48, 44)],
)
def test_one_pair_uses_pair_conditioned_chance_on_every_street(
    board, label_count, positive_count
):
    parent = pair_mass()
    state = ChanceState(board)
    children = tuple(propagate_joint_mass(state, parent))
    raw = chance_children(state)
    assert len(children) == chance_child_count(state) == label_count
    assert [(child.reveal, child.state) for child in children] == [
        (child.reveal, child.state) for child in raw
    ]
    totals = [child.joint_mass.total for child in children]
    assert totals.count(parent.total / positive_count) == positive_count
    assert totals.count(0) == label_count - positive_count
    assert sum(totals, Fraction()) == parent.total
    assert all(
        not child.reveal & (FIRST | SECOND)
        for child in children
        if child.joint_mass.total
    )


def test_correlated_mass_is_conserved_at_every_pair_coordinate():
    children = tuple(propagate_joint_mass(ChanceState(TURN), CORRELATED))
    recovered = add_masses(*(child.joint_mass for child in children))
    assert coordinates(recovered) == coordinates(CORRELATED)
    by_card = {child.reveal: child.joint_mass for child in children}
    # Ad blocks only the first pair. No cross-pairs may be invented by
    # reconstructing a product from the two marginals.
    assert coordinates(by_card[parse_card("Ad")]) == coordinates(
        pair_mass(OTHER_FIRST, OTHER_SECOND, Fraction(7, 5 * 44))
    )
    assert coordinates(by_card[parse_card("6s")]) == coordinates(CORRELATED.scaled(Fraction(1, 44)))


def test_propagation_is_linear_including_overlapping_support():
    left = CORRELATED
    right = pair_mass(weight=Fraction(2, 7))
    state = ChanceState(TURN)
    combined = propagate_joint_mass(state, add_masses(left.scaled(3), right.scaled(2)))
    for result, a, b in zip(
        combined, propagate_joint_mass(state, left), propagate_joint_mass(state, right),
        strict=True,
    ):
        assert coordinates(result.joint_mass) == coordinates(
            add_masses(a.joint_mass.scaled(3), b.joint_mass.scaled(2))
        )


def test_zero_mass_and_terminal_states_have_defined_behaviour():
    empty = JointMass(())
    children = tuple(propagate_joint_mass(ChanceState(TURN), empty))
    assert len(children) == 48
    assert all(child.joint_mass == empty for child in children)
    river = ChanceState(TURN | parse_card("6s"))
    assert chance_child_count(river) == 0
    assert tuple(propagate_joint_mass(river, CORRELATED)) == ()
    assert tuple(propagate_joint_mass(river, empty)) == ()


@pytest.mark.parametrize(
    "state,parent,message",
    [
        (ChanceState(TURN), pair_mass(first=make_hand(["2c", "Ah"])), "incompatible"),
        (ChanceState(TURN), pair_mass(second=make_hand(["Ad", "Kh"])), "incompatible"),
        (ChanceState(TURN, first=OTHER_FIRST), pair_mass(), "fixed hands"),
        (ChanceState(TURN, second=OTHER_SECOND), pair_mass(), "fixed hands"),
    ],
)
def test_invalid_input_is_rejected_without_silent_mass_loss(state, parent, message):
    with pytest.raises(ValueError, match=message):
        tuple(propagate_joint_mass(state, parent))


@pytest.mark.parametrize("fixed_second,expected_count", [(0, 46), (SECOND, 44)])
def test_fixed_hands_limit_labels_without_changing_pair_splits(fixed_second, expected_count):
    state = ChanceState(TURN, first=FIRST, second=fixed_second)
    children = tuple(propagate_joint_mass(state, pair_mass()))
    assert len(children) == expected_count
    assert sum(bool(child.joint_mass.total) for child in children) == 44
    assert all(child.state.first == FIRST for child in children)
    assert all(child.state.second == fixed_second for child in children)
    assert sum((child.joint_mass.total for child in children), Fraction()) == pair_mass().total


def test_core_does_not_normalize_materialize_dense_mass_or_evaluate_showdowns(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("mass propagation called a downstream operation")

    monkeypatch.setattr(JointMass, "normalized", forbidden)
    monkeypatch.setattr(JointMass, "to_dense", forbidden)
    monkeypatch.setattr(mass_module, "compare_hands", forbidden)
    iterator = propagate_joint_mass(ChanceState(TURN), CORRELATED)
    assert iter(iterator) is iterator
    assert len(tuple(iterator)) == 48


def test_reveal_order_preserves_final_mass_but_retains_different_histories():
    state = ChanceState(FLOP, reveals=(FLOP,), action_history=("check", "check"))
    first_card, second_card = parse_card("Js"), parse_card("6s")

    def follow(cards):
        current_state, current_mass = state, CORRELATED
        for card in cards:
            child = next(
                child for child in propagate_joint_mass(current_state, current_mass)
                if child.reveal == card
            )
            current_state, current_mass = child.state, child.joint_mass
        return current_state, current_mass

    forward_state, forward_mass = follow((first_card, second_card))
    reverse_state, reverse_mass = follow((second_card, first_card))
    assert forward_state.board == reverse_state.board
    assert forward_state.reveals == (FLOP, first_card, second_card)
    assert reverse_state.reveals == (FLOP, second_card, first_card)
    assert forward_state.action_history == reverse_state.action_history == state.action_history
    assert coordinates(forward_mass) == coordinates(reverse_mass)
    assert coordinates(forward_mass) == coordinates(CORRELATED.scaled(Fraction(1, 45 * 44)))


def test_propagation_commutes_with_suit_transport_of_asymmetric_mass():
    permutation = (2, 0, 3, 1)

    def move(joint_mass):
        return JointMass(tuple(
            JointEntry(
                HAND_INDEX[permute_cards(PRIVATE_HANDS[entry.first_index], permutation)],
                HAND_INDEX[permute_cards(PRIVATE_HANDS[entry.second_index], permutation)],
                entry.weight,
            )
            for entry in joint_mass.entries
        ))

    state = ChanceState(TURN, reveals=(FLOP, parse_card("Js")))
    moved = {
        child.reveal: child
        for child in propagate_joint_mass(permute_state(state, permutation), move(CORRELATED))
    }
    for child in propagate_joint_mass(state, CORRELATED):
        transported = moved[permute_cards(child.reveal, permutation)]
        assert transported.state == permute_state(child.state, permutation)
        assert coordinates(transported.joint_mass) == coordinates(move(child.joint_mass))
