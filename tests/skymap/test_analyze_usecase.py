# tests/skymap/test_analyze_usecase.py

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from radioskysims_skymap.application.skymap.analyze import (
    AnalyzeSkyMapUseCase,
)
from radioskysims_skymap.application.skymap.dto import (
    AnalyzeSkyMapInputDTO,
    AnalyzeSkyMapOutputDTO,
)
from radioskysims_skymap.application.skymap.ports import (
    SkyMapAnalyzerPort,
)
from radioskysims_skymap.domain.skymap import NSide, SkyMap


class FakeSkyMapAnalyzer(SkyMapAnalyzerPort):
    """Fake configurável para testar o caso de uso de análise."""

    def __init__(self, result: Any) -> None:
        self._result = result
        self.calls: list[dict[str, object]] = []

    def compute_cls(
        self,
        *,
        skymap: SkyMap,
        lmax: int | None = None,
    ) -> np.ndarray:
        self.calls.append(
            {
                "skymap": skymap,
                "lmax": lmax,
            }
        )

        return self._result


def make_skymap() -> SkyMap:
    """Cria um SkyMap válido para os testes."""
    return SkyMap(
        dataset=np.ones(12, dtype=np.float64),
        nside=NSide(1),
        unit="uK_CMB",
    )


def make_input_dto(
    *,
    sky_map: SkyMap | Any = None,
    lmax: int | None = None,
) -> AnalyzeSkyMapInputDTO:
    """Cria um DTO de entrada configurável."""
    if sky_map is None:
        sky_map = make_skymap()

    return AnalyzeSkyMapInputDTO(
        sky_map=sky_map,
        lmax=lmax,
    )


# ============================================================
# __init__()
# ============================================================


def test_init_rejects_invalid_analyzer() -> None:
    """O analyzer deve implementar SkyMapAnalyzerPort."""
    with pytest.raises(
        TypeError,
        match="analyzer deve implementar SkyMapAnalyzerPort",
    ):
        AnalyzeSkyMapUseCase(
            analyzer=object(),  # type: ignore[arg-type]
        )


# ============================================================
# execute() — caminho de sucesso
# ============================================================


def test_analyze_skymap_success() -> None:
    """O caso de uso deve retornar o espectro angular calculado."""
    sky_map = make_skymap()

    expected_cls = np.array(
        [1.0, 0.5, 0.25],
        dtype=np.float64,
    )

    analyzer = FakeSkyMapAnalyzer(
        result=expected_cls,
    )

    use_case = AnalyzeSkyMapUseCase(
        analyzer=analyzer,
    )

    input_dto = AnalyzeSkyMapInputDTO(
        sky_map=sky_map,
        lmax=2,
    )

    result = use_case.execute(input_dto)

    assert isinstance(result, AnalyzeSkyMapOutputDTO)
    assert result.nside == 1
    assert result.lmax == 2

    np.testing.assert_array_equal(
        result.cls,
        expected_cls,
    )


def test_execute_accepts_none_lmax() -> None:
    """lmax pode ser None."""
    expected_cls = np.ones(
        3,
        dtype=np.float64,
    )

    analyzer = FakeSkyMapAnalyzer(
        result=expected_cls,
    )

    use_case = AnalyzeSkyMapUseCase(
        analyzer=analyzer,
    )

    result = use_case.execute(
        AnalyzeSkyMapInputDTO(
            sky_map=make_skymap(),
            lmax=None,
        )
    )

    assert result.lmax is None

    np.testing.assert_array_equal(
        result.cls,
        expected_cls,
    )


def test_execute_delegates_correct_parameters() -> None:
    """O caso de uso deve encaminhar SkyMap e lmax à porta."""
    sky_map = make_skymap()

    analyzer = FakeSkyMapAnalyzer(
        result=np.ones(4, dtype=np.float64),
    )

    use_case = AnalyzeSkyMapUseCase(
        analyzer=analyzer,
    )

    use_case.execute(
        AnalyzeSkyMapInputDTO(
            sky_map=sky_map,
            lmax=3,
        )
    )

    assert len(analyzer.calls) == 1

    call = analyzer.calls[0]

    assert call["skymap"] is sky_map
    assert call["lmax"] == 3


# ============================================================
# execute() — validação do DTO
# ============================================================


def test_execute_rejects_none_input() -> None:
    """input_dto não pode ser None."""
    analyzer = FakeSkyMapAnalyzer(
        result=np.ones(3, dtype=np.float64),
    )

    use_case = AnalyzeSkyMapUseCase(
        analyzer=analyzer,
    )

    with pytest.raises(
        ValueError,
        match="input_dto não pode ser None",
    ):
        use_case.execute(None)  # type: ignore[arg-type]

    assert analyzer.calls == []


def test_execute_rejects_invalid_input_type() -> None:
    """input_dto deve ser AnalyzeSkyMapInputDTO."""
    analyzer = FakeSkyMapAnalyzer(
        result=np.ones(3, dtype=np.float64),
    )

    use_case = AnalyzeSkyMapUseCase(
        analyzer=analyzer,
    )

    with pytest.raises(
        TypeError,
        match="input_dto deve ser AnalyzeSkyMapInputDTO",
    ):
        use_case.execute(object())  # type: ignore[arg-type]

    assert analyzer.calls == []


def test_execute_rejects_none_skymap() -> None:
    """sky_map não pode ser None."""
    analyzer = FakeSkyMapAnalyzer(
        result=np.ones(3, dtype=np.float64),
    )

    use_case = AnalyzeSkyMapUseCase(
        analyzer=analyzer,
    )

    input_dto = AnalyzeSkyMapInputDTO(
        sky_map=None,  # type: ignore[arg-type]
        lmax=2,
    )

    with pytest.raises(
        ValueError,
        match="SkyMap não pode ser None",
    ):
        use_case.execute(input_dto)

    assert analyzer.calls == []


@pytest.mark.parametrize(
    "invalid_sky_map",
    [
        object(),
        np.ones(12, dtype=np.float64),
        "invalid",
    ],
)
def test_execute_rejects_invalid_skymap_type(
    invalid_sky_map: object,
) -> None:
    """sky_map deve ser uma entidade SkyMap."""
    analyzer = FakeSkyMapAnalyzer(
        result=np.ones(3, dtype=np.float64),
    )

    use_case = AnalyzeSkyMapUseCase(
        analyzer=analyzer,
    )

    input_dto = AnalyzeSkyMapInputDTO(
        sky_map=invalid_sky_map,  # type: ignore[arg-type]
        lmax=2,
    )

    with pytest.raises(
        TypeError,
        match="sky_map deve ser uma instância de SkyMap",
    ):
        use_case.execute(input_dto)

    assert analyzer.calls == []


@pytest.mark.parametrize(
    "lmax",
    [
        0,
        -1,
        -10,
    ],
)
def test_execute_rejects_non_positive_lmax(
    lmax: int,
) -> None:
    """lmax deve ser estritamente positivo."""
    analyzer = FakeSkyMapAnalyzer(
        result=np.ones(3, dtype=np.float64),
    )

    use_case = AnalyzeSkyMapUseCase(
        analyzer=analyzer,
    )

    input_dto = AnalyzeSkyMapInputDTO(
        sky_map=make_skymap(),
        lmax=lmax,
    )

    with pytest.raises(
        ValueError,
        match="lmax deve ser positivo",
    ):
        use_case.execute(input_dto)

    assert analyzer.calls == []


# ============================================================
# execute() — contrato da porta
# ============================================================


@pytest.mark.parametrize(
    "invalid_result",
    [
        None,
        object(),
        [1.0, 0.5, 0.25],
        (1.0, 0.5, 0.25),
    ],
)
def test_execute_rejects_invalid_analyzer_result(
    invalid_result: object,
) -> None:
    """A porta de análise deve retornar numpy.ndarray."""
    analyzer = FakeSkyMapAnalyzer(
        result=invalid_result,
    )

    use_case = AnalyzeSkyMapUseCase(
        analyzer=analyzer,
    )

    with pytest.raises(
        TypeError,
        match="SkyMapAnalyzerPort deve retornar numpy.ndarray",
    ):
        use_case.execute(
            AnalyzeSkyMapInputDTO(
                sky_map=make_skymap(),
                lmax=2,
            )
        )

    assert len(analyzer.calls) == 1
