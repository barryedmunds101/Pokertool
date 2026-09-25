import numpy as np

from representation import (
    GROUP_ORDER,
    PROJECTOR_NUMERATORS,
    ambient_character,
    irreducible_multiplicities,
    isotypic_dimensions,
    multiply_coefficients,
)


def test_ambient_character_is_derived_from_fixed_hands() -> None:
    assert ambient_character() == (1326, 338, 26, 78, 0)


def test_exact_irreducible_decomposition() -> None:
    assert irreducible_multiplicities() == {
        "[4]": 169,
        "[31]": 247,
        "[22]": 91,
        "[211]": 78,
        "[1111]": 0,
    }
    assert isotypic_dimensions() == {
        "[4]": 169,
        "[31]": 741,
        "[22]": 182,
        "[211]": 234,
        "[1111]": 0,
    }


def test_projector_numerators_are_exact_orthogonal_idempotents() -> None:
    projectors = tuple(PROJECTOR_NUMERATORS.values())
    zero = (0,) * GROUP_ORDER
    for left in projectors:
        for right in projectors:
            product = multiply_coefficients(left, right)
            expected = (
                tuple(GROUP_ORDER * coefficient for coefficient in left.coefficients)
                if left.irrep == right.irrep
                else zero
            )
            assert product == expected


def test_projector_numerators_resolve_24_times_identity() -> None:
    summed = tuple(
        sum(projector.coefficients[index] for projector in PROJECTOR_NUMERATORS.values())
        for index in range(GROUP_ORDER)
    )
    assert sorted(summed) == [0] * (GROUP_ORDER - 1) + [GROUP_ORDER]

    vector = np.arange(1326, dtype=np.int64)
    resolved = sum((projector.apply(vector) for projector in PROJECTOR_NUMERATORS.values()))
    assert np.array_equal(resolved, GROUP_ORDER * vector)
