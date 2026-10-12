from dataclasses import replace
from fractions import Fraction

import pytest

from exact_zero_sum import solve_zero_sum, verify_matrix_equilibrium


@pytest.mark.parametrize("matrix,value", [
    ([[1, -1], [-1, 1]], Fraction()),
    ([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], Fraction()),
    ([[1, -2], [-1, 1]], Fraction(-1, 5)),
    ([[0, 0, 0], [0, 0, 0]], Fraction()),
    ([[2, 2], [1, 1]], Fraction(2)),
    ([[Fraction(2, 7)]], Fraction(2, 7)),
    ([[3, 1, -2]], Fraction(-2)),
    ([[3], [1], [-2]], Fraction(3)),
])
def test_exact_matrix_games_have_verified_known_values(matrix, value):
    result = solve_zero_sum(matrix)
    assert result.value == value
    assert verify_matrix_equilibrium(matrix, result)
    assert all(isinstance(p, Fraction) for p in (*result.maximizer, *result.minimizer))


def test_indifference_equations_produce_fifths_without_a_grid():
    result = solve_zero_sum([[1, -2], [-1, 1]])
    assert result.maximizer == (Fraction(2, 5), Fraction(3, 5))
    assert result.minimizer == (Fraction(3, 5), Fraction(2, 5))


def test_three_way_mixing_and_false_certificate():
    matrix = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
    result = solve_zero_sum(matrix)
    assert result.maximizer == result.minimizer == (Fraction(1, 3),) * 3
    assert not verify_matrix_equilibrium(matrix, replace(result, value=Fraction(1, 100)))
    assert not verify_matrix_equilibrium(matrix, replace(result, maximizer=(Fraction(1), Fraction(), Fraction())))


@pytest.mark.parametrize("matrix,error", [
    ([], ValueError), ([[1], [1, 2]], ValueError), ([[0.5]], TypeError), ([[True]], TypeError),
])
def test_invalid_matrix_input_is_rejected(matrix, error):
    with pytest.raises(error):
        solve_zero_sum(matrix)


def test_pivot_limit_never_returns_an_uncertified_answer():
    with pytest.raises(RuntimeError, match="pivot limit"):
        solve_zero_sum([[1, -1], [-1, 1]], max_pivots=1)
