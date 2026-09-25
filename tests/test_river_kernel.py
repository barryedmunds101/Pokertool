import numpy as np

from cards import make_hand
from hand_space import HAND_INDEX
from river_kernel import result_matrix, river_kernel


def test_river_kernel_has_exact_algebraic_structure() -> None:
    board = make_hand(["2c", "7d", "9h", "Js", "3c"])
    compatibility, dominance = river_kernel(board)

    assert compatibility.shape == dominance.shape == (1326, 1326)
    assert compatibility.dtype == np.bool_
    assert dominance.dtype == np.int8
    assert np.array_equal(compatibility, compatibility.T)
    assert np.array_equal(dominance, -dominance.T)
    assert set(np.unique(dominance)) == {-1, 0, 1}
    assert np.all(dominance[~compatibility] == 0)


def test_kernel_encodes_loss_tie_win_and_incompatibility() -> None:
    board = make_hand(["2c", "7d", "9h", "Js", "3c"])
    compatibility, dominance = river_kernel(board)
    result = result_matrix(board)
    loser = HAND_INDEX[make_hand(["Tc", "Ad"])]
    winner = HAND_INDEX[make_hand(["9c", "Kd"])]
    tied_first = HAND_INDEX[make_hand(["4c", "5d"])]
    tied_second = HAND_INDEX[make_hand(["4d", "5c"])]
    blocked = HAND_INDEX[make_hand(["2c", "Kh"])]

    assert (compatibility[loser, winner], dominance[loser, winner], result[loser, winner]) == (True, -1, 0)
    assert (compatibility[winner, loser], dominance[winner, loser], result[winner, loser]) == (True, 1, 2)
    assert (compatibility[tied_first, tied_second], dominance[tied_first, tied_second], result[tied_first, tied_second]) == (True, 0, 1)
    assert not compatibility[loser, loser]
    assert result[loser, loser] == 0
    assert not compatibility[blocked].any()
    assert not dominance[blocked].any()
    assert not compatibility[:, blocked].any()
    assert not dominance[:, blocked].any()
