# tests/skymap/test_continuum_survey_usecase.py

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from radioskysims_skymap.application.skymap.generate import (
    ContinuumSurveyMapUseCase,
)
from radioskysims_skymap.application.skymap.ports import (
    ContinuumSurveyMapPort,
)
from radioskysims_skymap.domain.skymap import NSide, SkyMap
from radioskysims_skymap.domain.skymap.exceptions import (
    InvalidNSideError,
)


class FakeContinuumSurveyMapProvider(ContinuumSurveyMapPort):
    """Fake configurável para testar o caso de uso de surveys."""

    def __init__(self, result: Any) -> None:
        self._result = result
        self.calls: list[dict[str, object]] = []

    def fetch_cutout(
        self,
        *,
        survey_id: str,
        ra_deg: float,
        dec_deg: float,
        fov_deg: float,
        nside: NSide,
        freq_ghz: float | None = None,
        field: str = "I",
    ) -> SkyMap:
        self.calls.append(
            {
                "survey_id": survey_id,
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
                "fov_deg": fov_deg,
                "nside": nside,
                "freq_ghz": freq_ghz,
                "field": field,
            }
        )

        return self._result


def make_skymap(
    *,
    freq_ghz: float | None = 1.4,
    field: str = "I",
) -> SkyMap:
    """Cria um SkyMap válido para os testes."""
    return SkyMap(
        dataset=np.ones(12, dtype=np.float64),
        nside=NSide(1),
        unit="Jy/beam",
        frame="equatorial",
        freq_ghz=freq_ghz,
        field=field,
    )


def make_use_case(
    *,
    result: Any | None = None,
) -> tuple[
    ContinuumSurveyMapUseCase,
    FakeContinuumSurveyMapProvider,
]:
    """Cria caso de uso e provider fake."""
    if result is None:
        result = make_skymap()

    provider = FakeContinuumSurveyMapProvider(
        result=result,
    )

    return (
        ContinuumSurveyMapUseCase(provider=provider),
        provider,
    )


# ============================================================
# __init__()
# ============================================================


def test_init_rejects_invalid_provider() -> None:
    """Provider deve implementar ContinuumSurveyMapPort."""
    with pytest.raises(
        TypeError,
        match="provider deve implementar ContinuumSurveyMapPort",
    ):
        ContinuumSurveyMapUseCase(
            provider=object(),  # type: ignore[arg-type]
        )


# ============================================================
# execute() — caminho de sucesso
# ============================================================


def test_fetch_continuum_survey_success() -> None:
    """Deve retornar o SkyMap fornecido pelo provider."""
    expected = make_skymap(
        freq_ghz=1.4,
        field="I",
    )

    provider = FakeContinuumSurveyMapProvider(
        result=expected,
    )

    use_case = ContinuumSurveyMapUseCase(
        provider=provider,
    )

    result = use_case.execute(
        survey_id="NVSS",
        ra_deg=180.0,
        dec_deg=-30.0,
        fov_deg=5.0,
        nside=1,
        freq_ghz=1.4,
        field="I",
    )

    assert result is expected
    assert isinstance(result, SkyMap)


def test_execute_delegates_correct_parameters() -> None:
    """Deve encaminhar todos os parâmetros corretamente."""
    use_case, provider = make_use_case()

    use_case.execute(
        survey_id="ivo://CSIRO/NVSS",
        ra_deg=120.5,
        dec_deg=-45.25,
        fov_deg=2.5,
        nside=1,
        freq_ghz=1.4,
        field="Q",
    )

    assert len(provider.calls) == 1

    call = provider.calls[0]

    assert call["survey_id"] == "ivo://CSIRO/NVSS"
    assert call["ra_deg"] == 120.5
    assert call["dec_deg"] == -45.25
    assert call["fov_deg"] == 2.5
    assert call["nside"] == NSide(1)
    assert call["freq_ghz"] == 1.4
    assert call["field"] == "Q"


def test_execute_accepts_none_frequency() -> None:
    """freq_ghz pode ser None."""
    expected = make_skymap(
        freq_ghz=None,
    )

    provider = FakeContinuumSurveyMapProvider(
        result=expected,
    )

    use_case = ContinuumSurveyMapUseCase(
        provider=provider,
    )

    result = use_case.execute(
        survey_id="NVSS",
        ra_deg=0.0,
        dec_deg=0.0,
        fov_deg=1.0,
        nside=1,
        freq_ghz=None,
    )

    assert result is expected
    assert provider.calls[0]["freq_ghz"] is None
    assert provider.calls[0]["field"] == "I"


@pytest.mark.parametrize(
    "field",
    ["I", "Q", "U"],
)
def test_execute_accepts_supported_fields(
    field: str,
) -> None:
    """Deve aceitar todos os campos de Stokes suportados."""
    expected = make_skymap(field=field)

    provider = FakeContinuumSurveyMapProvider(
        result=expected,
    )

    use_case = ContinuumSurveyMapUseCase(
        provider=provider,
    )

    result = use_case.execute(
        survey_id="NVSS",
        ra_deg=180.0,
        dec_deg=0.0,
        fov_deg=1.0,
        nside=1,
        field=field,
    )

    assert result.field == field
    assert provider.calls[0]["field"] == field


@pytest.mark.parametrize(
    ("ra_deg", "dec_deg"),
    [
        (0.0, -90.0),
        (0.0, 90.0),
        (359.999, 0.0),
    ],
)
def test_execute_accepts_coordinate_boundaries(
    ra_deg: float,
    dec_deg: float,
) -> None:
    """Deve aceitar coordenadas nos limites permitidos."""
    use_case, provider = make_use_case()

    use_case.execute(
        survey_id="NVSS",
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        fov_deg=1.0,
        nside=1,
    )

    assert len(provider.calls) == 1


# ============================================================
# execute() — survey_id
# ============================================================


@pytest.mark.parametrize(
    "survey_id",
    [
        "",
        " ",
        "\t",
        "\n",
    ],
)
def test_execute_rejects_empty_survey_id(
    survey_id: str,
) -> None:
    """survey_id não pode ser vazio ou conter apenas espaços."""
    use_case, provider = make_use_case()

    with pytest.raises(
        ValueError,
        match="survey_id não pode ser vazio",
    ):
        use_case.execute(
            survey_id=survey_id,
            ra_deg=180.0,
            dec_deg=0.0,
            fov_deg=1.0,
            nside=1,
        )

    assert provider.calls == []


# ============================================================
# execute() — coordenadas
# ============================================================


@pytest.mark.parametrize(
    "ra_deg",
    [
        -0.001,
        -1.0,
        360.0,
        720.0,
    ],
)
def test_execute_rejects_invalid_ra(
    ra_deg: float,
) -> None:
    """ra_deg deve pertencer ao intervalo [0, 360)."""
    use_case, provider = make_use_case()

    with pytest.raises(
        ValueError,
        match="ra_deg deve estar entre 0 e 360 graus",
    ):
        use_case.execute(
            survey_id="NVSS",
            ra_deg=ra_deg,
            dec_deg=0.0,
            fov_deg=1.0,
            nside=1,
        )

    assert provider.calls == []


@pytest.mark.parametrize(
    "dec_deg",
    [
        -90.001,
        -180.0,
        90.001,
        180.0,
    ],
)
def test_execute_rejects_invalid_dec(
    dec_deg: float,
) -> None:
    """dec_deg deve pertencer ao intervalo [-90, 90]."""
    use_case, provider = make_use_case()

    with pytest.raises(
        ValueError,
        match="dec_deg deve estar entre -90 e 90 graus",
    ):
        use_case.execute(
            survey_id="NVSS",
            ra_deg=180.0,
            dec_deg=dec_deg,
            fov_deg=1.0,
            nside=1,
        )

    assert provider.calls == []


# ============================================================
# execute() — parâmetros físicos
# ============================================================


@pytest.mark.parametrize(
    "fov_deg",
    [
        0.0,
        -0.1,
        -10.0,
    ],
)
def test_execute_rejects_non_positive_fov(
    fov_deg: float,
) -> None:
    """fov_deg deve ser estritamente positivo."""
    use_case, provider = make_use_case()

    with pytest.raises(
        ValueError,
        match="fov_deg deve ser positivo",
    ):
        use_case.execute(
            survey_id="NVSS",
            ra_deg=180.0,
            dec_deg=0.0,
            fov_deg=fov_deg,
            nside=1,
        )

    assert provider.calls == []


@pytest.mark.parametrize(
    "nside",
    [
        0,
        -1,
        -8,
    ],
)
def test_execute_rejects_non_positive_nside(
    nside: int,
) -> None:
    """nside deve ser estritamente positivo."""
    use_case, provider = make_use_case()

    with pytest.raises(
        ValueError,
        match="nside deve ser positivo",
    ):
        use_case.execute(
            survey_id="NVSS",
            ra_deg=180.0,
            dec_deg=0.0,
            fov_deg=1.0,
            nside=nside,
        )

    assert provider.calls == []


def test_execute_rejects_invalid_healpix_nside() -> None:
    """NSide positivo também deve ser válido para HEALPix."""
    use_case, provider = make_use_case()

    with pytest.raises(InvalidNSideError):
        use_case.execute(
            survey_id="NVSS",
            ra_deg=180.0,
            dec_deg=0.0,
            fov_deg=1.0,
            nside=3,
        )

    assert provider.calls == []


@pytest.mark.parametrize(
    "freq_ghz",
    [
        0.0,
        -0.1,
        -100.0,
    ],
)
def test_execute_rejects_non_positive_frequency(
    freq_ghz: float,
) -> None:
    """Frequência fornecida deve ser estritamente positiva."""
    use_case, provider = make_use_case()

    with pytest.raises(
        ValueError,
        match="freq_ghz deve ser positiva",
    ):
        use_case.execute(
            survey_id="NVSS",
            ra_deg=180.0,
            dec_deg=0.0,
            fov_deg=1.0,
            nside=1,
            freq_ghz=freq_ghz,
        )

    assert provider.calls == []


@pytest.mark.parametrize(
    "field",
    [
        "",
        "V",
        "i",
        "QU",
    ],
)
def test_execute_rejects_invalid_field(
    field: str,
) -> None:
    """field deve ser I, Q ou U."""
    use_case, provider = make_use_case()

    with pytest.raises(
        ValueError,
        match="field deve ser 'I', 'Q' ou 'U'",
    ):
        use_case.execute(
            survey_id="NVSS",
            ra_deg=180.0,
            dec_deg=0.0,
            fov_deg=1.0,
            nside=1,
            field=field,
        )

    assert provider.calls == []


# ============================================================
# execute() — contrato da porta
# ============================================================


@pytest.mark.parametrize(
    "invalid_result",
    [
        None,
        object(),
        np.ones(12, dtype=np.float64),
        [1.0, 2.0, 3.0],
    ],
)
def test_execute_rejects_invalid_provider_result(
    invalid_result: object,
) -> None:
    """ContinuumSurveyMapPort deve retornar uma entidade SkyMap."""
    provider = FakeContinuumSurveyMapProvider(
        result=invalid_result,
    )

    use_case = ContinuumSurveyMapUseCase(
        provider=provider,
    )

    with pytest.raises(
        TypeError,
        match="ContinuumSurveyMapPort deve retornar SkyMap",
    ):
        use_case.execute(
            survey_id="NVSS",
            ra_deg=180.0,
            dec_deg=0.0,
            fov_deg=1.0,
            nside=1,
        )

    assert len(provider.calls) == 1
