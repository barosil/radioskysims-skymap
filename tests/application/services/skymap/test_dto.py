# tests/application/skymap/test_dto.py

from __future__ import annotations

from dataclasses import FrozenInstanceError

import healpy as hp
import numpy as np
import pytest

from radioskysims_skymap.application.skymap.dto import (
    AnalyzeSkyMapInputDTO,
    AnalyzeSkyMapOutputDTO,
    GenerateSkyMapInputDTO,
    GenerateSkyMapOutputDTO,
)
from radioskysims_skymap.domain.skymap.models import SkyMap
from radioskysims_skymap.domain.skymap.value_objects import NSide

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_skymap() -> SkyMap:
    nside = NSide(16)
    data = np.random.randn(hp.nside2npix(nside.value))

    return SkyMap(
        dataset=data,
        nside=nside,
        unit="uK_CMB",
        frame="galactic",
        freq_ghz=30.0,
        field="I",
    )


# =============================================================================
# GenerateSkyMapInputDTO
# =============================================================================


def test_generate_input_creation():
    dto = GenerateSkyMapInputDTO(
        frequency_ghz=30.0,
        nside=32,
        components=("cmb", "dust"),
    )

    assert dto.frequency_ghz == 30.0
    assert dto.nside == 32
    assert dto.components == ("cmb", "dust")


def test_generate_input_equality():
    dto1 = GenerateSkyMapInputDTO(
        30.0,
        32,
        ("cmb",),
    )

    dto2 = GenerateSkyMapInputDTO(
        30.0,
        32,
        ("cmb",),
    )

    assert dto1 == dto2


def test_generate_input_frozen():
    dto = GenerateSkyMapInputDTO(
        30.0,
        32,
        ("cmb",),
    )

    with pytest.raises(FrozenInstanceError):
        dto.nside = 64


# =============================================================================
# GenerateSkyMapOutputDTO
# =============================================================================


def test_generate_output_creation():
    data = np.arange(12)

    dto = GenerateSkyMapOutputDTO(
        nside=1,
        npix=12,
        map_data=data,
    )

    assert dto.nside == 1
    assert dto.npix == 12
    assert np.array_equal(dto.map_data, data)


def test_generate_output_frozen():
    dto = GenerateSkyMapOutputDTO(
        nside=1,
        npix=12,
        map_data=np.arange(12),
    )

    with pytest.raises(FrozenInstanceError):
        dto.npix = 24


# =============================================================================
# AnalyzeSkyMapInputDTO
# =============================================================================


def test_analyze_input_default_lmax(sample_skymap):
    dto = AnalyzeSkyMapInputDTO(
        sky_map=sample_skymap,
    )

    assert dto.sky_map is sample_skymap
    assert dto.lmax is None


def test_analyze_input_custom_lmax(sample_skymap):
    dto = AnalyzeSkyMapInputDTO(
        sky_map=sample_skymap,
        lmax=128,
    )

    assert dto.lmax == 128


def test_analyze_input_frozen(sample_skymap):
    dto = AnalyzeSkyMapInputDTO(sample_skymap)

    with pytest.raises(FrozenInstanceError):
        dto.lmax = 256


# =============================================================================
# AnalyzeSkyMapOutputDTO
# =============================================================================


def test_analyze_output_creation():
    cls = np.arange(11, dtype=float)

    dto = AnalyzeSkyMapOutputDTO(
        nside=16,
        lmax=10,
        cls=cls,
    )

    assert dto.nside == 16
    assert dto.lmax == 10
    assert np.array_equal(dto.cls, cls)


def test_analyze_output_equality():
    cls = np.arange(11)

    dto1 = AnalyzeSkyMapOutputDTO(
        16,
        10,
        cls,
    )

    dto2 = AnalyzeSkyMapOutputDTO(
        16,
        10,
        cls.copy(),
    )

    assert dto1.nside == dto2.nside
    assert dto1.lmax == dto2.lmax
    assert np.array_equal(dto1.cls, dto2.cls)


def test_analyze_output_frozen():
    dto = AnalyzeSkyMapOutputDTO(
        16,
        10,
        np.arange(11),
    )

    with pytest.raises(FrozenInstanceError):
        dto.nside = 32
