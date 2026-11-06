from __future__ import annotations

from unittest.mock import patch

import healpy as hp
import numpy as np
import pytest
from radioskysims_skymap.domain.skymap.factory import SkyMapFactory
from radioskysims_skymap.domain.skymap.models import SkyMap
from radioskysims_skymap.domain.skymap.value_objects import NSide

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def nside():
    return NSide(16)


@pytest.fixture
def dataset():
    return np.random.randn(hp.nside2npix(16))


@pytest.fixture
def factory():
    return SkyMapFactory(
        unit="uK_CMB",
        frame="galactic",
        field="I",
    )


# =============================================================================
# Construction
# =============================================================================


def test_create_skymap(factory, dataset, nside):
    sky = factory.create(
        dataset=dataset,
        nside=nside,
        freq_ghz=70.0,
    )

    assert isinstance(sky, SkyMap)

    assert sky.unit == "uK_CMB"
    assert sky.frame == "galactic"
    assert sky.field == "I"
    assert sky.freq_ghz == 70.0


# =============================================================================
# Override metadata
# =============================================================================


def test_create_override_metadata(factory, dataset, nside):
    sky = factory.create(
        dataset=dataset,
        nside=nside,
        freq_ghz=30.0,
        unit="Jy",
        frame="equatorial",
        field="Q",
    )

    assert sky.unit == "Jy"
    assert sky.frame == "equatorial"
    assert sky.field == "Q"
    assert sky.freq_ghz == 30.0


# =============================================================================
# Default metadata
# =============================================================================


def test_factory_default_values(dataset, nside):
    factory = SkyMapFactory()

    sky = factory.create(
        dataset=dataset,
        nside=nside,
    )

    assert sky.unit == "unknown"
    assert sky.frame == "galactic"
    assert sky.field == "I"


# =============================================================================
# Exception handling
# =============================================================================


def test_create_reraises_exception():
    # B2: o domínio é puro — SkyMapFactory não logga; a exceção é propagada.
    factory = SkyMapFactory()

    with (
        patch(
            "radioskysims_skymap.domain.skymap.factory.SkyMap",
            side_effect=RuntimeError("boom"),
        ),
        pytest.raises(RuntimeError, match="boom"),
    ):
        factory.create(
            dataset=np.zeros(12),
            nside=NSide(1),
        )
