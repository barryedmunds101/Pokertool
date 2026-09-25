import numpy as np

from cards import make_hand, parse_card
from hand_space import compatible_hand_indices
from river_kernel import river_kernel
from symmetry import (
    HAND_PERMUTATIONS,
    SUIT_PERMUTATIONS,
    board_orbit_count,
    canonical_board,
    compose,
    hand_permutation,
    inverse,
    permute_card,
    permute_cards,
    private_hand_orbits,
    stabilizer,
)


IDENTITY = (0, 1, 2, 3)


def test_s4_elements_composition_and_inverses() -> None:
    assert len(SUIT_PERMUTATIONS) == len(set(SUIT_PERMUTATIONS)) == 24
    assert len(HAND_PERMUTATIONS) == 24
    for permutation in SUIT_PERMUTATIONS:
        assert compose(permutation, inverse(permutation)) == IDENTITY
        assert compose(inverse(permutation), permutation) == IDENTITY
        indices = hand_permutation(permutation)
        assert np.array_equal(np.sort(indices), np.arange(1326))


def test_card_and_card_set_actions_agree() -> None:
    permutation = (2, 0, 3, 1)
    cards = make_hand(["Ac", "Td", "4h", "7s"])
    expected = sum(permute_card(card, permutation) for card in (
        parse_card("Ac"), parse_card("Td"), parse_card("4h"), parse_card("7s")
    ))
    assert permute_cards(cards, permutation) == expected


def test_known_suit_orbit_counts() -> None:
    assert len(private_hand_orbits()) == 169
    assert board_orbit_count(5) == 134459


def test_canonical_board_and_monotone_stabilizer() -> None:
    board = make_hand(["2c", "5c", "8c", "Jc", "Ac"])
    assert len(stabilizer(board)) == 6
    canonical = canonical_board(board)
    assert all(canonical <= permute_cards(board, g) for g in SUIT_PERMUTATIONS)
    assert all(canonical_board(permute_cards(board, g)) == canonical for g in SUIT_PERMUTATIONS)


def test_river_kernel_is_equivariant_under_suit_permutations() -> None:
    board = make_hand(["2c", "7d", "9h", "Js", "3c"])
    permutation = (2, 0, 3, 1)
    moved_board = permute_cards(board, permutation)
    indices = hand_permutation(permutation)
    compatibility, dominance = river_kernel(board)
    moved_compatibility, moved_dominance = river_kernel(moved_board)

    coordinates = np.ix_(indices, indices)
    assert np.array_equal(moved_compatibility[coordinates], compatibility)
    assert np.array_equal(moved_dominance[coordinates], dominance)


def test_monotone_board_kernel_commutes_with_its_stabilizer() -> None:
    board = make_hand(["2c", "5c", "8c", "Jc", "Ac"])
    compatibility, dominance = river_kernel(board)
    compatible = set(compatible_hand_indices(board))
    assert len(compatible) == 1081

    for permutation in stabilizer(board):
        indices = hand_permutation(permutation)
        coordinates = np.ix_(indices, indices)
        assert {int(indices[index]) for index in compatible} == compatible
        assert np.array_equal(compatibility[coordinates], compatibility)
        assert np.array_equal(dominance[coordinates], dominance)
