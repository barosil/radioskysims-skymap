# tests/application/skymap/test_generate.py

from __future__ import annotations

import healpy as hp
import numpy as np
import pytest
from radioskysims_skymap.domain.skymap.exceptions import InvalidNSideError

from radioskysims_skymap.application.skymap.dto import GenerateSkyMapInputDTO
from radioskysims_skymap.application.skymap.generate import (
    ContinuumSurveyMapUseCase,
    GenerateSkyMapUseCase,
)
from radioskysims_skymap.application.skymap.ports import (
    ContinuumSurveyMapPort,
    SkyMapGeneratorPort,
)
from radioskysims_skymap.domain.skymap.models import SkyMap
from radioskysims_skymap.domain.skymap.value_objects import NSide


# =============================================================================
# Dummy Generator
# =============================================================================


class DummyGenerator(SkyMapGeneratorPort):
    def __init__(self):
        self.called = False
        self.args = None

    def generate(
        self,
        *,
        nside: NSide,
        freq_ghz: float,
        components=None,
    ) -> SkyMap:

        self.called = True
        self.args = (nside, freq_ghz, components)

        return SkyMap(
            dataset=np.zeros(hp.nside2npix(nside.value)),
            nside=nside,
            unit="uK_CMB",
            freq_ghz=freq_ghz,
        )


# =============================================================================
# Dummy Survey Provider
# =============================================================================


class DummySurveyProvider(ContinuumSurveyMapPort):
    def __init__(self):
        self.called = False
        self.kwargs = None

    def fetch_cutout(
        self,
        *,
        survey_id,
        ra_deg,
        dec_deg,
        fov_deg,
        nside,
        freq_ghz=None,
        field="I",
    ):

        self.called = True

        self.kwargs = {
            "survey_id": survey_id,
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
            "fov_deg": fov_deg,
            "nside": nside,
            "freq_ghz": freq_ghz,
            "field": field,
        }

        return SkyMap(
            dataset=np.zeros(hp.nside2npix(nside.value)),
            nside=nside,
            unit="Jy",
            freq_ghz=freq_ghz,
            field=field,
        )


# =============================================================================
# GenerateSkyMapUseCase
# =============================================================================


def test_generate_success():

    generator = DummyGenerator()

    usecase = GenerateSkyMapUseCase(generator)

    dto = GenerateSkyMapInputDTO(
        frequency_ghz=30.0,
        nside=16,
        components=("cmb", "dust"),
    )

    result = usecase.execute(dto)

    assert generator.called
    assert result.nside == 16
    assert result.npix == hp.nside2npix(16)
    assert result.map_data.shape == (hp.nside2npix(16),)


def test_generator_receives_correct_parameters():

    generator = DummyGenerator()

    usecase = GenerateSkyMapUseCase(generator)

    dto = GenerateSkyMapInputDTO(
        frequency_ghz=70.0,
        nside=32,
        components=("cmb",),
    )

    usecase.execute(dto)

    nside, freq, comps = generator.args

    assert nside == NSide(32)
    assert freq == 70.0
    assert comps == ("cmb",)


@pytest.mark.parametrize(
    "freq",
    [
        0,
        -1,
        -20,
    ],
)
def test_invalid_frequency(freq):

    generator = DummyGenerator()

    usecase = GenerateSkyMapUseCase(generator)

    dto = GenerateSkyMapInputDTO(
        frequency_ghz=freq,
        nside=16,
        components=("cmb",),
    )

    with pytest.raises(ValueError):
        usecase.execute(dto)


@pytest.mark.parametrize(
    "components",
    [
        (),
        [],
        None,
    ],
)
def test_empty_components(components):

    generator = DummyGenerator()

    usecase = GenerateSkyMapUseCase(generator)

    dto = GenerateSkyMapInputDTO(
        frequency_ghz=30.0,
        nside=16,
        components=components,
    )

    with pytest.raises(ValueError):
        usecase.execute(dto)


def test_invalid_nside():

    generator = DummyGenerator()

    usecase = GenerateSkyMapUseCase(generator)

    dto = GenerateSkyMapInputDTO(
        frequency_ghz=30,
        nside=3,
        components=("cmb",),
    )

    with pytest.raises(InvalidNSideError):
        usecase.execute(dto)


# =============================================================================
# ContinuumSurveyMapUseCase
# =============================================================================


def test_continuum_usecase():

    provider = DummySurveyProvider()

    usecase = ContinuumSurveyMapUseCase(provider)

    result = usecase.execute(
        survey_id="haslam408",
        ra_deg=180,
        dec_deg=0,
        fov_deg=10,
        nside=16,
        freq_ghz=0.408,
    )

    assert provider.called
    assert isinstance(result, SkyMap)
    assert result.nside_value == 16
    assert result.freq_ghz == 0.408


def test_continuum_parameters():

    provider = DummySurveyProvider()

    usecase = ContinuumSurveyMapUseCase(provider)

    usecase.execute(
        survey_id="survey",
        ra_deg=15,
        dec_deg=-22,
        fov_deg=8,
        nside=32,
        freq_ghz=1.4,
        field="Q",
    )

    args = provider.kwargs

    assert args["survey_id"] == "survey"
    assert args["ra_deg"] == 15
    assert args["dec_deg"] == -22
    assert args["fov_deg"] == 8
    assert args["nside"] == NSide(32)
    assert args["freq_ghz"] == 1.4
    assert args["field"] == "Q"


def test_continuum_invalid_nside():

    provider = DummySurveyProvider()

    usecase = ContinuumSurveyMapUseCase(provider)

    with pytest.raises(InvalidNSideError):
        usecase.execute(
            survey_id="survey",
            ra_deg=0,
            dec_deg=0,
            fov_deg=5,
            nside=5,
        )


def test_generator_exception_is_propagated():

    class BrokenGenerator(DummyGenerator):
        def generate(self, **_kwargs):
            msg = "generator failed"
            raise RuntimeError(msg)

    usecase = GenerateSkyMapUseCase(BrokenGenerator())

    dto = GenerateSkyMapInputDTO(
        frequency_ghz=30,
        nside=16,
        components=("cmb",),
    )

    with pytest.raises(RuntimeError):
        usecase.execute(dto)


def test_provider_exception_is_propagated():

    class BrokenProvider(ContinuumSurveyMapPort):
        def fetch_cutout(self, **_kwargs):
            msg = "provider failed"
            raise RuntimeError(msg)

    usecase = ContinuumSurveyMapUseCase(BrokenProvider())

    with pytest.raises(RuntimeError):
        usecase.execute(
            survey_id="survey",
            ra_deg=0,
            dec_deg=0,
            fov_deg=5,
            nside=16,
        )
