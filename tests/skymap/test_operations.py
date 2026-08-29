
# tests/skymap/test_operations.py

from __future__ import annotations

from unittest.mock import ANY, patch

import numpy as np
import pytest

from radioskysims_skymap.domain.skymap.exceptions import (
    AnalysisError,
    InvalidNSideError,
)
from radioskysims_skymap.domain.skymap.operations import SkyMapOperations
from radioskysims_skymap.domain.skymap.value_objects import NSide


# ============================================================
# smooth()
# ============================================================


def test_smooth_preserves_shape() -> None:
    """A suavização deve preservar o tamanho do mapa."""
    map_data = np.ones(12, dtype=np.float64)

    result = SkyMapOperations.smooth(
        map_data,
        np.radians(1.0),
    )

    assert result.shape == map_data.shape


def test_smooth_invalid_map() -> None:
    """map_data deve ser numpy.ndarray."""
    with pytest.raises(AnalysisError):
        SkyMapOperations.smooth(
            "not_an_array",
            np.radians(1.0),
        )


@pytest.mark.parametrize(
    "fwhm",
    [0.0, -1.0, -0.5],
)
def test_smooth_invalid_fwhm(fwhm: float) -> None:
    """FWHM deve ser positivo."""
    map_data = np.ones(12, dtype=np.float64)

    with pytest.raises(AnalysisError):
        SkyMapOperations.smooth(
            map_data,
            fwhm,
        )


# ============================================================
# rotate()
# ============================================================


def test_rotate_invalid_rotation() -> None:
    """A rotação deve possuir exatamente três ângulos."""
    map_data = np.ones(12, dtype=np.float64)

    with pytest.raises(AnalysisError):
        SkyMapOperations.rotate(
            map_data,
            (0.0, 0.0),
        )


@patch("radioskysims_skymap.domain.skymap.operations.hp")
def test_rotate_success(mock_hp) -> None:
    """Verifica a sequência de chamadas do Healpy."""
    map_data = np.ones(12, dtype=np.float64)

    alm = np.array(
        [1.0 + 0.0j, 2.0 + 0.0j],
        dtype=np.complex128,
    )

    expected = np.ones(12, dtype=np.float64)

    mock_hp.get_nside.return_value = 1
    mock_hp.map2alm.return_value = alm
    mock_hp.alm2map.return_value = expected

    result = SkyMapOperations.rotate(
        map_data,
        (0.0, 0.0, 0.0),
    )

    mock_hp.get_nside.assert_called_once_with(map_data)
    mock_hp.map2alm.assert_called_once_with(map_data)

    mock_hp.rotate_alm.assert_called_once_with(
        ANY,
        rot=(0.0, 0.0, 0.0),
    )

    mock_hp.alm2map.assert_called_once_with(
        ANY,
        nside=1,
        verbose=False,
    )

    rotated_alm = mock_hp.rotate_alm.call_args.args[0]

    assert rotated_alm is not alm

    np.testing.assert_array_equal(
        rotated_alm,
        alm,
    )

    np.testing.assert_array_equal(
        result,
        expected,
    )


# ============================================================
# downgrade()
# ============================================================


def test_downgrade_changes_nside() -> None:
    """O downgrade deve reduzir o número de pixels."""
    map_data = np.arange(
        48.0,
        dtype=np.float64,
    )

    result = SkyMapOperations.downgrade(
        map_data,
        NSide(1),
    )

    assert result.shape == (12,)


def test_downgrade_invalid_nside() -> None:
    """target_nside deve ser um Value Object NSide."""
    map_data = np.arange(
        48.0,
        dtype=np.float64,
    )

    with pytest.raises(InvalidNSideError):
        SkyMapOperations.downgrade(
            map_data,
            1,
        )
