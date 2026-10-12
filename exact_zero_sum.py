"""A small rational LP solver for finite two-player zero-sum matrix games.

This is a reference solver, not a large-poker implementation. A positive payoff
shift gives a packing LP with an initially feasible slack basis. Bland's pivot
rule and Fraction arithmetic avoid probability grids and numerical tolerances.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class MatrixEquilibrium:
    maximizer: tuple[Fraction, ...]
    minimizer: tuple[Fraction, ...]
    value: Fraction
    pivots: int = 0


def _matrix(payoffs: Sequence[Sequence[int | Fraction]]) -> tuple[tuple[Fraction, ...], ...]:
    if not payoffs or not payoffs[0]:
        raise ValueError("payoff matrix must be nonempty")
    width = len(payoffs[0])
    rows = []
    for row in payoffs:
        if len(row) != width:
            raise ValueError("payoff matrix must be rectangular")
        if any(isinstance(value, bool) or not isinstance(value, (int, Fraction)) for value in row):
            raise TypeError("payoffs must be integers or Fractions")
        rows.append(tuple(Fraction(value) for value in row))
    return tuple(rows)


def verify_matrix_equilibrium(
    payoffs: Sequence[Sequence[int | Fraction]], solution: MatrixEquilibrium
) -> bool:
    """Check both finite families of deviation inequalities without solving."""
    matrix = _matrix(payoffs)
    x, y = solution.maximizer, solution.minimizer
    if len(x) != len(matrix) or len(y) != len(matrix[0]):
        return False
    if any(isinstance(p, bool) or not isinstance(p, (int, Fraction)) or p < 0 for p in (*x, *y)):
        return False
    if not isinstance(solution.value, (int, Fraction)) or isinstance(solution.value, bool):
        return False
    if sum(x, Fraction()) != 1 or sum(y, Fraction()) != 1:
        return False
    row_values = [sum((a * p for a, p in zip(row, y)), Fraction()) for row in matrix]
    column_values = [
        sum((x[i] * matrix[i][j] for i in range(len(x))), Fraction())
        for j in range(len(y))
    ]
    return max(row_values) <= solution.value <= min(column_values)


def solve_zero_sum(
    payoffs: Sequence[Sequence[int | Fraction]], *, max_pivots: int = 10_000
) -> MatrixEquilibrium:
    """Solve a small matrix game exactly and verify its saddle inequalities.

    For M = A + shift > 0, maximize sum(z) subject to Mz <= 1, z >= 0.
    The dual is minimize sum(w), M^T w >= 1, w >= 0. At optimum S,
    x = w/S, y = z/S, and the original game value is 1/S - shift.
    """
    if isinstance(max_pivots, bool) or not isinstance(max_pivots, int) or max_pivots < 1:
        raise ValueError("max_pivots must be a positive integer")
    matrix = _matrix(payoffs)
    m, n = len(matrix), len(matrix[0])
    smallest = min(min(row) for row in matrix)
    shift = max(Fraction(), 1 - smallest)
    tableau = [
        [value + shift for value in row]
        + [Fraction(i == j) for j in range(m)]
        + [Fraction(1)]
        for i, row in enumerate(matrix)
    ]
    tableau.append([Fraction(-1)] * n + [Fraction()] * (m + 1))
    basis = [n + i for i in range(m)]
    pivots = 0

    while True:
        entering = next((j for j, value in enumerate(tableau[-1][:-1]) if value < 0), None)
        if entering is None:
            break
        candidates = [i for i in range(m) if tableau[i][entering] > 0]
        if not candidates:
            raise RuntimeError("positive-payoff packing LP unexpectedly became unbounded")
        leaving = min(candidates, key=lambda i: (tableau[i][-1] / tableau[i][entering], basis[i]))
        if pivots >= max_pivots:
            raise RuntimeError("exact LP pivot limit reached; no equilibrium is being certified")
        pivot = tableau[leaving][entering]
        tableau[leaving] = [value / pivot for value in tableau[leaving]]
        for i in range(m + 1):
            if i == leaving:
                continue
            factor = tableau[i][entering]
            if factor:
                tableau[i] = [a - factor * b for a, b in zip(tableau[i], tableau[leaving])]
        basis[leaving] = entering
        pivots += 1

    primal = [Fraction()] * n
    for i, variable in enumerate(basis):
        if variable < n:
            primal[variable] = tableau[i][-1]
    dual = tableau[-1][n : n + m]
    total = tableau[-1][-1]
    if total <= 0:
        raise RuntimeError("packing LP has no positive normalization")
    result = MatrixEquilibrium(
        tuple(value / total for value in dual),
        tuple(value / total for value in primal),
        1 / total - shift,
        pivots,
    )
    if not verify_matrix_equilibrium(matrix, result):
        raise RuntimeError("exact LP result failed the independent saddle-inequality check")
    return result
