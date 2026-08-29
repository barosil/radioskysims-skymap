# tests/infrastructure/skymap/test_utils.py

import numpy as np
import pytest

from radioskysims_skymap.infrastructure.skymap.utils import (
    expected_npix,
    validate_healpix_map,
    mask_invalid,
)
from radioskysims_skymap.domain.skymap.value_objects import NSide


def test_expected_npix():
    assert expected_npix(NSide(1)) == 12
    assert expected_npix(2) == 48


def test_validate_healpix_map_ok():
    data = np.ones(12)
    validate_healpix_map(data, NSide(1))


def test_validate_healpix_map_wrong_size():
    data = np.ones(10)
    with pytest.raises(ValueError):
        validate_healpix_map(data, NSide(1))


def test_mask_invalid():
    data = np.array([1.0, np.nan, np.inf])
    cleaned = mask_invalid(data)
    assert np.all(np.isfinite(cleaned))
