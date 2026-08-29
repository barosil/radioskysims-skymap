from __future__ import annotations

import healpy as hp
import numpy as np

from radioskysims_skymap.domain.skymap.factory import SkyMapFactory
from radioskysims_skymap.domain.skymap.value_objects import NSide


def test_generate_returns_skymap():
    nside = NSide(32)
    factory = SkyMapFactory(unit="uK_CMB")
    data = np.ones(hp.nside2npix(nside.value))

    skymap = factory.create(dataset=data, nside=nside, freq_ghz=30.0)

    assert skymap.nside == nside


def test_generated_map_is_numpy_array():
    nside = NSide(32)
    factory = SkyMapFactory(unit="uK_CMB")
    data = np.ones(hp.nside2npix(nside.value))

    skymap = factory.create(dataset=data, nside=nside, freq_ghz=30.0)

    assert isinstance(skymap.data, np.ndarray)
    assert skymap.data.size == hp.nside2npix(nside.value)
