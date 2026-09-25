"""Exact representation theory for the ambient 1,326-hand suit action.

The central idempotents are stored as integer numerator operators

    Q_lambda = d_lambda sum_g chi_lambda(g) rho(g),

so that the corresponding rational projector is ``Q_lambda / 24``.
No dense 1,326 by 1,326 permutation or projector matrices are retained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from symmetry import HAND_PERMUTATIONS, SUIT_PERMUTATIONS, SuitPermutation, compose


GROUP_ORDER: Final = len(SUIT_PERMUTATIONS)
IDENTITY: Final[SuitPermutation] = (0, 1, 2, 3)

# Conjugacy classes are named by cycle partition, in the conventional S4 order.
CLASS_ORDER: Final = ((1, 1, 1, 1), (2, 1, 1), (2, 2), (3, 1), (4,))
CLASS_SIZES: Final = (1, 6, 3, 8, 6)

# Integer characters of the five irreducibles of S4.
CHARACTER_TABLE: Final = {
    "[4]": (1, 1, 1, 1, 1),
    "[31]": (3, 1, -1, 0, -1),
    "[22]": (2, 0, 2, -1, 0),
    "[211]": (3, -1, -1, 0, 1),
    "[1111]": (1, -1, 1, 1, -1),
}


def cycle_type(permutation: SuitPermutation) -> tuple[int, ...]:
    """Return the decreasing cycle partition of a suit permutation."""
    unseen = set(range(4))
    lengths: list[int] = []
    while unseen:
        start = min(unseen)
        current = start
        length = 0
        while current in unseen:
            unseen.remove(current)
            length += 1
            current = permutation[current]
        lengths.append(length)
    return tuple(sorted(lengths, reverse=True))


CLASS_INDEX: Final = {partition: index for index, partition in enumerate(CLASS_ORDER)}


def ambient_character() -> tuple[int, ...]:
    """Derive the hand representation character from fixed coordinates."""
    values: list[int | None] = [None] * len(CLASS_ORDER)
    for permutation, hand_indices in zip(SUIT_PERMUTATIONS, HAND_PERMUTATIONS):
        index = CLASS_INDEX[cycle_type(permutation)]
        fixed = int(np.count_nonzero(hand_indices == np.arange(hand_indices.size)))
        if values[index] is not None and values[index] != fixed:
            raise AssertionError("character is not constant on a conjugacy class")
        values[index] = fixed
    return tuple(int(value) for value in values if value is not None)


def irreducible_multiplicities(
    character: tuple[int, ...] | None = None,
) -> dict[str, int]:
    """Decompose a class character using exact character inner products."""
    character = ambient_character() if character is None else character
    if len(character) != len(CLASS_ORDER):
        raise ValueError("character must have one value for each S4 conjugacy class")
    result: dict[str, int] = {}
    for name, irreducible in CHARACTER_TABLE.items():
        numerator = sum(
            size * value * irrep_value
            for size, value, irrep_value in zip(CLASS_SIZES, character, irreducible)
        )
        if numerator % GROUP_ORDER:
            raise ValueError("character does not have integral irreducible multiplicities")
        result[name] = numerator // GROUP_ORDER
    return result


@dataclass(frozen=True)
class ProjectorNumerator:
    """A central projector numerator represented in the integral group algebra."""

    irrep: str
    coefficients: tuple[int, ...]

    def apply(self, values: np.ndarray) -> np.ndarray:
        """Apply ``Q_lambda`` exactly along the first (hand-coordinate) axis."""
        array = np.asarray(values)
        if array.ndim == 0 or array.shape[0] != len(HAND_PERMUTATIONS[0]):
            raise ValueError("values must have 1,326 entries on its first axis")
        dtype = np.result_type(array.dtype, np.int64)
        result = np.zeros(array.shape, dtype=dtype)
        for coefficient, indices in zip(self.coefficients, HAND_PERMUTATIONS):
            if coefficient:
                result[indices] += coefficient * array
        return result

    def dense(self) -> np.ndarray:
        """Materialize the integer matrix, primarily for small-scale inspection."""
        dimension = len(HAND_PERMUTATIONS[0])
        result = np.zeros((dimension, dimension), dtype=np.int16)
        columns = np.arange(dimension)
        for coefficient, indices in zip(self.coefficients, HAND_PERMUTATIONS):
            if coefficient:
                result[indices, columns] += coefficient
        return result


def projector_numerator(irrep: str) -> ProjectorNumerator:
    """Construct the exact numerator ``Q_lambda`` for a named S4 irrep."""
    try:
        character = CHARACTER_TABLE[irrep]
    except KeyError:
        raise ValueError(f"unknown S4 irrep: {irrep!r}") from None
    dimension = character[0]
    coefficients = tuple(
        dimension * character[CLASS_INDEX[cycle_type(permutation)]]
        for permutation in SUIT_PERMUTATIONS
    )
    return ProjectorNumerator(irrep, coefficients)


PROJECTOR_NUMERATORS: Final = {
    name: projector_numerator(name) for name in CHARACTER_TABLE
}


def multiply_coefficients(
    left: ProjectorNumerator, right: ProjectorNumerator
) -> tuple[int, ...]:
    """Multiply two numerator operators exactly in the group algebra."""
    positions = {permutation: index for index, permutation in enumerate(SUIT_PERMUTATIONS)}
    result = [0] * GROUP_ORDER
    for first, first_coefficient in zip(SUIT_PERMUTATIONS, left.coefficients):
        for second, second_coefficient in zip(SUIT_PERMUTATIONS, right.coefficients):
            result[positions[compose(first, second)]] += first_coefficient * second_coefficient
    return tuple(result)


def isotypic_dimensions() -> dict[str, int]:
    """Return dimensions of the five ambient isotypic components."""
    multiplicities = irreducible_multiplicities()
    return {
        name: multiplicities[name] * character[0]
        for name, character in CHARACTER_TABLE.items()
    }
