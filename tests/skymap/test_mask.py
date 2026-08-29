# tests/skymap/test_mask.py

from __future__ import annotations
import numpy as np
import pytest

from radioskysims_skymap.domain.skymap.mask import SkyMask


# ============================================================
# apply() — caminho de sucesso
# ============================================================


def test_apply_mask_ok() -> None:
    """Deve multiplicar o mapa pela máscara."""
    map_data = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float64,
    )
    mask = np.array(
        [1.0, 0.0, 1.0],
        dtype=np.float64,
    )

    result = SkyMask.apply(
        map_data,
        mask,
    )

    expected = np.array(
        [1.0, 0.0, 3.0],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(
        result,
        expected,
    )


def test_apply_weighted_mask() -> None:
    """Deve aceitar máscaras ponderadas, não apenas binárias."""
    map_data = np.array(
        [2.0, 4.0, 8.0],
        dtype=np.float64,
    )
    mask = np.array(
        [0.5, 0.25, 0.125],
        dtype=np.float64,
    )

    result = SkyMask.apply(
        map_data,
        mask,
    )

    expected = np.array(
        [1.0, 1.0, 1.0],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_apply_unit_mask() -> None:
    """Uma máscara unitária deve preservar o mapa."""
    map_data = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float64,
    )
    mask = np.ones_like(map_data)

    result = SkyMask.apply(
        map_data,
        mask,
    )

    np.testing.assert_array_equal(
        result,
        map_data,
    )


def test_apply_zero_mask() -> None:
    """Uma máscara nula deve produzir um mapa nulo."""
    map_data = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float64,
    )
    mask = np.zeros_like(map_data)

    result = SkyMask.apply(
        map_data,
        mask,
    )

    np.testing.assert_array_equal(
        result,
        np.zeros_like(map_data),
    )


def test_apply_returns_float64() -> None:
    """O resultado deve ser normalizado para float64."""
    map_data = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float32,
    )
    mask = np.array(
        [1.0, 0.5, 0.0],
        dtype=np.float32,
    )

    result = SkyMask.apply(
        map_data,
        mask,
    )

    assert result.dtype == np.float64


def test_apply_does_not_modify_inputs() -> None:
    """A operação não deve modificar mapa ou máscara."""
    map_data = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float64,
    )
    mask = np.array(
        [1.0, 0.0, 1.0],
        dtype=np.float64,
    )

    original_map = map_data.copy()
    original_mask = mask.copy()

    result = SkyMask.apply(
        map_data,
        mask,
    )

    assert result is not map_data
    assert result is not mask

    np.testing.assert_array_equal(
        map_data,
        original_map,
    )
    np.testing.assert_array_equal(
        mask,
        original_mask,
    )


# ============================================================
# _validate() — tipos
# ============================================================


@pytest.mark.parametrize(
    "invalid_map",
    [
        None,
        "invalid",
        [1.0, 2.0],
        (1.0, 2.0),
        1,
    ],
)
def test_invalid_map_type(
    invalid_map: object,
) -> None:
    """map_data deve ser numpy.ndarray."""
    mask = np.ones(2, dtype=np.float64)

    with pytest.raises(
        TypeError,
        match="map_data deve ser numpy.ndarray",
    ):
        SkyMask.apply(
            invalid_map,  # type: ignore[arg-type]
            mask,
        )


@pytest.mark.parametrize(
    "invalid_mask",
    [
        None,
        "invalid",
        [1.0, 0.0],
        (1.0, 0.0),
        1,
    ],
)
def test_invalid_mask_type(
    invalid_mask: object,
) -> None:
    """mask deve ser numpy.ndarray."""
    map_data = np.ones(2, dtype=np.float64)

    with pytest.raises(
        TypeError,
        match="mask deve ser numpy.ndarray",
    ):
        SkyMask.apply(
            map_data,
            invalid_mask,  # type: ignore[arg-type]
        )


# ============================================================
# _validate() — dimensionalidade e shape
# ============================================================


def test_map_data_must_be_one_dimensional() -> None:
    """map_data deve possuir exatamente uma dimensão."""
    map_data = np.ones(
        (2, 2),
        dtype=np.float64,
    )
    mask = np.ones(
        4,
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="map_data deve ser unidimensional",
    ):
        SkyMask.apply(
            map_data,
            mask,
        )


def test_mask_must_be_one_dimensional() -> None:
    """mask deve possuir exatamente uma dimensão."""
    map_data = np.ones(
        4,
        dtype=np.float64,
    )
    mask = np.ones(
        (2, 2),
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="mask deve ser unidimensional",
    ):
        SkyMask.apply(
            map_data,
            mask,
        )


def test_mask_shape_incompatible() -> None:
    """Mapa e máscara devem possuir o mesmo shape."""
    map_data = np.ones(
        12,
        dtype=np.float64,
    )
    mask = np.ones(
        10,
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="map_data e mask devem possuir o mesmo shape",
    ):
        SkyMask.apply(
            map_data,
            mask,
        )


# ============================================================
# _validate() — conteúdo
# ============================================================


def test_empty_map_is_invalid() -> None:
    """Mapa e máscara vazios devem ser rejeitados."""
    map_data = np.array(
        [],
        dtype=np.float64,
    )
    mask = np.array(
        [],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="map_data não pode estar vazio",
    ):
        SkyMask.apply(
            map_data,
            mask,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_map_data_must_contain_only_finite_values(
    invalid_value: float,
) -> None:
    """map_data não deve conter NaN ou infinito."""
    map_data = np.array(
        [1.0, invalid_value, 3.0],
        dtype=np.float64,
    )
    mask = np.ones(
        3,
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="map_data contém valores não finitos",
    ):
        SkyMask.apply(
            map_data,
            mask,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_mask_must_contain_only_finite_values(
    invalid_value: float,
) -> None:
    """mask não deve conter NaN ou infinito."""
    map_data = np.ones(
        3,
        dtype=np.float64,
    )
    mask = np.array(
        [1.0, invalid_value, 0.0],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="mask contém valores não finitos",
    ):
        SkyMask.apply(
            map_data,
            mask,
        )
