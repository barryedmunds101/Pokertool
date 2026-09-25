import pytest

from cards import make_hand
from hand_space import HAND_INDEX, PRIVATE_HANDS, compatible_hand_indices, hand_label


def test_ambient_private_hand_basis_is_stable_and_indexed() -> None:
    assert len(PRIVATE_HANDS) == 1326
    assert len(HAND_INDEX) == 1326
    assert all(HAND_INDEX[hand] == index for index, hand in enumerate(PRIVATE_HANDS))


def test_compatible_indices_preserve_ambient_coordinates() -> None:
    board = make_hand(["As", "Kd", "Qh", "Jc", "2s"])
    indices = compatible_hand_indices(board)

    assert len(indices) == 1081
    assert all(PRIVATE_HANDS[index] & board == 0 for index in indices)
    assert HAND_INDEX[make_hand(["Ac", "Ad"])] in indices
    assert HAND_INDEX[make_hand(["As", "Ad"])] not in indices


def test_hand_labels_and_validation() -> None:
    assert hand_label(make_hand(["2c", "3c"])) == "2c3c"
    with pytest.raises(ValueError, match="two-card"):
        hand_label(make_hand(["As"]))
