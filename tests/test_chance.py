from fractions import Fraction

from cards import make_hand
from chance import (
    ChanceState,
    canonical_chance_children,
    canonical_state,
    chance_children,
    permute_state,
)
from symmetry import permute_cards


def hand(*names: str) -> int:
    return make_hand(names)


FIRST = hand("Ah", "Kh")
SECOND = hand("Js", "Jd")


def test_each_street_returns_only_immediate_children() -> None:
    preflop = ChanceState(first=FIRST, second=SECOND, action_history=("raise", "call"))
    flops = chance_children(preflop)
    assert len(flops) == 17296
    assert all(child.state.street == "flop" for child in flops)
    assert all(child.probability == Fraction(1, 17296) for child in flops)
    assert all(child.state.action_history == ("raise", "call") for child in flops)
    assert sum((child.probability for child in flops), Fraction()) == 1

    flop = flops[0].state
    turns = chance_children(flop)
    assert len(turns) == 45
    assert all(child.state.street == "turn" for child in turns)
    assert all(child.probability == Fraction(1, 45) for child in turns)

    turn = turns[0].state
    rivers = chance_children(turn)
    assert len(rivers) == 44
    assert all(child.state.street == "river" for child in rivers)
    assert all(child.probability == Fraction(1, 44) for child in rivers)
    assert chance_children(rivers[0].state) == ()


def test_public_and_conditioned_turn_have_different_probabilities() -> None:
    board = hand("2c", "5c", "8c", "Jc")
    public = chance_children(ChanceState(board))
    conditioned = chance_children(ChanceState(board, FIRST, SECOND))

    assert len(public) == 48
    assert len(conditioned) == 44
    assert public[0].probability == Fraction(1, 48)
    assert conditioned[0].probability == Fraction(1, 44)


def test_canonical_key_is_invariant_and_transport_is_explicit() -> None:
    state = ChanceState(
        hand("2c", "7d", "9h", "Qs"),
        FIRST,
        SECOND,
        action_history=("check", "bet", "call"),
    )
    permutation = (2, 0, 3, 1)
    moved = permute_state(state, permutation)
    key, transport = canonical_state(state)
    moved_key, _ = canonical_state(moved)

    assert key == moved_key
    assert canonical_state(permute_state(state, transport))[0] == key
    assert permute_state(state, transport).action_history == state.action_history


def test_children_group_only_under_parent_stabilizer() -> None:
    state = ChanceState(hand("2c", "5c", "8c", "Jc"))
    raw = chance_children(state)
    grouped = canonical_chance_children(state)

    assert len(raw) == 48
    assert len(grouped) == 22
    assert sorted(group.multiplicity for group in grouped) == [1] * 9 + [3] * 13
    assert sum(group.multiplicity for group in grouped) == 48
    assert sum((group.probability for group in grouped), Fraction()) == 1
    for group in grouped:
        assert len(group.members) == group.multiplicity
        for member in group.members:
            assert (
                permute_state(member.child.state, member.permutation)
                == group.representative.state
            )


def test_fixed_blockers_are_part_of_conditional_symmetry() -> None:
    state = ChanceState(hand("2c", "5c", "8c", "Jc"), FIRST, SECOND)
    grouped = canonical_chance_children(state)

    assert sum(group.multiplicity for group in grouped) == 44
    assert sum((group.probability for group in grouped), Fraction()) == 1
    for group in grouped:
        for member in group.members:
            moved = permute_state(member.child.state, member.permutation)
            assert moved.first == group.representative.state.first
            assert moved.second == group.representative.state.second
