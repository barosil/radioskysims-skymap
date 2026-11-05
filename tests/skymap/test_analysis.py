from __future__ import annotations

from unittest.mock import MagicMock, patch

import healpy as hp
import numpy as np
import pytest

from radioskysims_skymap.domain.skymap.analysis import SkyMapAnalysis


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def nside():
    return 16


@pytest.fixture
def npix(nside):
    return hp.nside2npix(nside)


@pytest.fixture
def zero_map(npix):
    return np.zeros(npix)


@pytest.fixture
def unit_map(npix):
    return np.ones(npix)


@pytest.fixture
def random_map(npix):
    np.random.seed(42)
    return np.random.randn(npix)


# =============================================================================
# Mean
# =============================================================================


def test_mean_zero_map(zero_map):
    """A média de um mapa nulo deve ser zero."""
    assert SkyMapAnalysis.mean(zero_map) == 0.0


def test_mean_invalid_type():
    """mean() deve aceitar apenas numpy.ndarray."""
    with pytest.raises(TypeError):
        SkyMapAnalysis.mean([1, 2, 3])


# =============================================================================
# RMS
# =============================================================================


def test_rms_unit_map(unit_map):
    """O RMS de um mapa unitário deve ser 1."""
    assert SkyMapAnalysis.rms(unit_map) == 1.0


def test_rms_invalid_type():
    with pytest.raises(TypeError):
        SkyMapAnalysis.rms([1, 2, 3])


# =============================================================================
# Min / Max
# =============================================================================


def test_min_max():
    m = np.array([-2.0, 0.0, 3.0])

    assert SkyMapAnalysis.min_max(m) == (-2.0, 3.0)


def test_min_max_invalid_type():
    with pytest.raises(TypeError):
        SkyMapAnalysis.min_max([1, 2, 3])


# =============================================================================
# Angular Power Spectrum
# =============================================================================


def test_angular_power_spectrum(random_map):
    """
    Teste funcional usando Healpy.
    """

    cl = SkyMapAnalysis.angular_power_spectrum(
        random_map,
        lmax=10,
    )

    assert len(cl) == 11


def test_angular_power_invalid_map():
    """
    map_data deve ser numpy.ndarray.
    """

    with pytest.raises(TypeError):
        SkyMapAnalysis.angular_power_spectrum(
            [1, 2, 3],
            lmax=10,
        )


@pytest.mark.parametrize(
    "lmax",
    [
        0,
        -1,
        -10,
    ],
)
def test_angular_power_invalid_lmax(random_map, lmax):
    """
    lmax deve ser positivo.
    """

    with pytest.raises(ValueError):
        SkyMapAnalysis.angular_power_spectrum(
            random_map,
            lmax=lmax,
        )


@patch("radioskysims_skymap.domain.skymap.analysis.importlib.import_module")
def test_angular_power_without_healpy(mock_import, random_map):
    """
    Simula ausência do Healpy.
    """

    mock_import.side_effect = ImportError

    with pytest.raises(ImportError):
        SkyMapAnalysis.angular_power_spectrum(
            random_map,
            lmax=10,
        )


@patch("radioskysims_skymap.domain.skymap.analysis.importlib.import_module")
def test_angular_power_imports_healpy(mock_import, random_map):
    """
    Verifica que o módulo healpy é carregado dinamicamente.
    """

    fake_hp = MagicMock()

    fake_hp.anafast.return_value = np.arange(11)

    mock_import.return_value = fake_hp

    result = SkyMapAnalysis.angular_power_spectrum(
        random_map,
        lmax=10,
    )

    mock_import.assert_called_once_with("healpy")

    fake_hp.anafast.assert_called_once()

    assert len(result) == 11
