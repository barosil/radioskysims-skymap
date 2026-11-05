# tests/application/skymap/test_ports.py

from __future__ import annotations

from collections.abc import Sequence

import healpy as hp
import numpy as np
import pytest

from radioskysims_skymap.application.skymap.ports import (
    ContinuumSurveyMapPort,
    SkyMapAnalyzerPort,
    SkyMapGeneratorPort,
    SkyMapMaskPort,
)
from radioskysims_skymap.domain.skymap.models import SkyMap
from radioskysims_skymap.domain.skymap.value_objects import NSide


# =============================================================================
# Fixture
# =============================================================================


@pytest.fixture
def sample_skymap() -> SkyMap:
    nside = NSide(16)

    return SkyMap(
        dataset=np.zeros(hp.nside2npix(nside.value)),
        nside=nside,
        unit="uK_CMB",
    )


# =============================================================================
# Dummy implementations
# =============================================================================


class DummyGenerator(SkyMapGeneratorPort):
    def generate(
        self,
        *,
        nside: NSide,
        freq_ghz: float,
        components: Sequence[str] | None = None,  # noqa: ARG002
    ) -> SkyMap:
        return SkyMap(
            dataset=np.zeros(hp.nside2npix(nside.value)),
            nside=nside,
            unit="uK_CMB",
            freq_ghz=freq_ghz,
        )


class DummyAnalyzer(SkyMapAnalyzerPort):
    def compute_cls(self, skymap: SkyMap) -> object:  # noqa: ARG002
        return {"cls": np.arange(5)}


class DummyMask(SkyMapMaskPort):
    def apply(self, skymap: SkyMap, mask: SkyMap) -> SkyMap:  # noqa: ARG002
        return skymap


class DummySurveyProvider(ContinuumSurveyMapPort):
    def fetch_cutout(
        self,
        *,
        survey_id: str,  # noqa: ARG002
        ra_deg: float,  # noqa: ARG002
        dec_deg: float,  # noqa: ARG002
        fov_deg: float,  # noqa: ARG002
        nside: NSide,
        freq_ghz: float | None = None,
        field: str = "I",
    ) -> SkyMap:
        return SkyMap(
            dataset=np.zeros(hp.nside2npix(nside.value)),
            nside=nside,
            unit="Jy",
            freq_ghz=freq_ghz,
            field=field,
        )


# =============================================================================
# Abstract classes
# =============================================================================


def test_generator_is_abstract():
    with pytest.raises(TypeError):
        SkyMapGeneratorPort()


def test_analyzer_is_abstract():
    with pytest.raises(TypeError):
        SkyMapAnalyzerPort()


def test_mask_is_abstract():
    with pytest.raises(TypeError):
        SkyMapMaskPort()


def test_continuum_provider_is_abstract():
    with pytest.raises(TypeError):
        ContinuumSurveyMapPort()


# =============================================================================
# Generator
# =============================================================================


def test_generator_contract():
    gen = DummyGenerator()

    result = gen.generate(
        nside=NSide(16),
        freq_ghz=30.0,
        components=("cmb",),
    )

    assert isinstance(result, SkyMap)
    assert result.nside_value == 16
    assert result.freq_ghz == 30.0


# =============================================================================
# Analyzer
# =============================================================================


def test_analyzer_contract(sample_skymap):
    analyzer = DummyAnalyzer()

    result = analyzer.compute_cls(sample_skymap)

    assert isinstance(result, dict)
    assert "cls" in result


# =============================================================================
# Mask
# =============================================================================


def test_mask_contract(sample_skymap):
    masker = DummyMask()

    result = masker.apply(sample_skymap, sample_skymap)

    assert isinstance(result, SkyMap)


# =============================================================================
# ContinuumSurveyMapPort
# =============================================================================


def test_continuum_provider_contract():
    provider = DummySurveyProvider()

    result = provider.fetch_cutout(
        survey_id="haslam408",
        ra_deg=180.0,
        dec_deg=0.0,
        fov_deg=10.0,
        nside=NSide(16),
        freq_ghz=0.408,
    )

    assert isinstance(result, SkyMap)
    assert result.nside_value == 16
    assert result.freq_ghz == 0.408
