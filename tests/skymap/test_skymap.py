from __future__ import annotations

from unittest.mock import patch

import healpy as hp
import numpy as np
import pytest

from radioskysims_skymap.domain.skymap.exceptions import InvalidDatasetError
from radioskysims_skymap.domain.skymap.models import SkyMap
from radioskysims_skymap.domain.skymap.value_objects import NSide


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def nside():
    return NSide(32)


@pytest.fixture
def npix():
    return hp.nside2npix(32)


@pytest.fixture
def dataset(npix):
    return np.random.randn(npix)


@pytest.fixture
def skymap(dataset):
    return SkyMap(
        dataset=dataset,
        nside=32,
        unit="uK_CMB",
        frame="galactic",
        freq_ghz=70.0,
    )


# =============================================================================
# Construction
# =============================================================================


def test_create_skymap_smoke(dataset, npix):
    """
    Smoke test equivalente ao teste original.
    """

    m = SkyMap(
        dataset=dataset,
        nside=32,
        unit="uK_CMB",
        frame="galactic",
        freq_ghz=70.0,
    )

    assert m.nside.value == 32
    assert m.npix == npix
    assert m.unit == "uK_CMB"
    assert m.frame == "galactic"
    assert m.field == "I"


def test_constructor_accepts_nside_object(dataset):
    m = SkyMap(
        dataset=dataset,
        nside=NSide(32),
        unit="K",
    )

    assert m.nside.value == 32


def test_constructor_accepts_integer_nside(dataset):
    m = SkyMap(
        dataset=dataset,
        nside=32,
        unit="K",
    )

    assert isinstance(m.nside, NSide)


# =============================================================================
# Validation
# =============================================================================


def test_invalid_empty_unit(dataset):
    with pytest.raises(InvalidDatasetError):
        SkyMap(
            dataset=dataset,
            nside=32,
            unit="",
        )


@pytest.mark.parametrize(
    "frame",
    [
        "",
        "fk5",
        "mars",
        "invalid",
    ],
)
def test_invalid_frame(dataset, frame):
    with pytest.raises(InvalidDatasetError):
        SkyMap(
            dataset=dataset,
            nside=32,
            unit="K",
            frame=frame,
        )


@pytest.mark.parametrize(
    "field",
    [
        "",
        "A",
        "temperature",
        "V",
    ],
)
def test_invalid_field(dataset, field):
    with pytest.raises(InvalidDatasetError):
        SkyMap(
            dataset=dataset,
            nside=32,
            unit="K",
            field=field,
        )


def test_invalid_dataset_dimension():
    data = np.zeros((3, 4))

    with pytest.raises(InvalidDatasetError):
        SkyMap(
            dataset=data,
            nside=1,
            unit="K",
        )


def test_invalid_dataset_size():
    """
    48 pixels não correspondem ao NSide=1.
    """
    data = np.zeros(48)

    with pytest.raises(InvalidDatasetError):
        SkyMap(
            dataset=data,
            nside=1,
            unit="K",
        )


# =============================================================================
# Properties
# =============================================================================


def test_data_returns_copy(skymap):
    data = skymap.data

    data[0] = 999

    assert skymap.data[0] != 999


def test_nside_property(skymap):
    assert isinstance(skymap.nside, NSide)


def test_nside_value_property(skymap):
    assert skymap.nside_value == 32


def test_npix_property(skymap):
    assert skymap.npix == hp.nside2npix(32)


# =============================================================================
# Copy
# =============================================================================


def test_copy_returns_new_instance(skymap):
    clone = skymap.copy()

    assert clone is not skymap
    assert np.array_equal(clone.data, skymap.data)


def test_copy_preserves_metadata(skymap):
    clone = skymap.copy()

    assert clone.unit == skymap.unit
    assert clone.frame == skymap.frame
    assert clone.field == skymap.field
    assert clone.freq_ghz == skymap.freq_ghz


def test_copy_is_independent(skymap):
    clone = skymap.copy()

    copied = clone.data

    copied[0] = 999

    assert skymap.data[0] != 999


# =============================================================================
# Mask
# =============================================================================


def test_apply_mask(skymap):
    mask = np.zeros(skymap.npix)

    masked = skymap.apply_mask(mask)

    assert np.all(masked.data == 0)


def test_apply_mask_invalid_type(skymap):
    with pytest.raises(TypeError):
        skymap.apply_mask([1] * skymap.npix)


def test_apply_mask_invalid_shape(skymap):
    with pytest.raises(ValueError):
        skymap.apply_mask(np.ones(10))


# =============================================================================
# Factory methods
# =============================================================================


def test_from_array(dataset):
    sky = SkyMap.from_array(
        dataset,
        NSide(32),
        unit="K",
    )

    assert isinstance(sky, SkyMap)
    assert sky.npix == hp.nside2npix(32)


@patch("radioskysims_skymap.domain.skymap.models.hp.map2alm")
def test_alm(mock_map2alm, skymap):
    expected = np.array([1.0, 2.0])

    mock_map2alm.return_value = expected

    result = skymap.alm()

    mock_map2alm.assert_called_once()

    assert np.array_equal(result, expected)


@patch("radioskysims_skymap.domain.skymap.models.hp.alm2map")
def test_from_alm(mock_alm2map):
    mock_alm2map.return_value = np.ones(hp.nside2npix(32))

    sky = SkyMap.from_alm(
        alm=np.array([1.0]),
        nside=NSide(32),
        unit="K",
    )

    assert isinstance(sky, SkyMap)
    assert sky.npix == hp.nside2npix(32)

    mock_alm2map.assert_called_once()
