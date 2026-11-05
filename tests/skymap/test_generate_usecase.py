# tests/skymap/test_generate_usecase.py

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from radioskysims_skymap.application.skymap.dto import (
    GenerateSkyMapInputDTO,
    GenerateSkyMapOutputDTO,
)
from radioskysims_skymap.application.skymap.generate import (
    GenerateSkyMapUseCase,
)
from radioskysims_skymap.application.skymap.ports import (
    SkyMapGeneratorPort,
)
from radioskysims_skymap.domain.skymap.models import SkyMap
from radioskysims_skymap.domain.skymap.value_objects import NSide


class FakeSkyMapGenerator(SkyMapGeneratorPort):
    """Fake configurável para os testes do caso de uso."""

    def __init__(
        self,
        result: SkyMap | Any,
    ) -> None:
        self._result = result
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        *,
        nside: NSide,
        freq_ghz: float,
        components: Sequence[str] | None = None,
    ) -> SkyMap:
        self.calls.append(
            {
                "nside": nside,
                "freq_ghz": freq_ghz,
                "components": components,
            }
        )

        return self._result


def make_skymap() -> SkyMap:
    """Cria um SkyMap válido usado pelos testes."""
    return SkyMap(
        dataset=np.ones(12, dtype=np.float64),
        nside=NSide(1),
        unit="uK_CMB",
    )


def make_input_dto(
    *,
    nside: int = 1,
    frequency_ghz: float = 100.0,
    components: tuple[object, ...] = ("cmb",),
) -> GenerateSkyMapInputDTO:
    """
    Cria o DTO de entrada.

    O tipo de components é propositalmente amplo para permitir
    testar valores inválidos em tempo de execução.
    """
    return GenerateSkyMapInputDTO(
        nside=nside,
        frequency_ghz=frequency_ghz,
        components=components,  # type: ignore[arg-type]
    )


# ============================================================
# __init__()
# ============================================================


def test_init_rejects_invalid_generator() -> None:
    """O gerador deve implementar SkyMapGeneratorPort."""
    invalid_generator = object()

    with pytest.raises(
        TypeError,
        match="generator deve implementar SkyMapGeneratorPort",
    ):
        GenerateSkyMapUseCase(
            generator=invalid_generator,  # type: ignore[arg-type]
        )


# ============================================================
# execute() — caminho de sucesso
# ============================================================


def test_generate_skymap_success() -> None:
    """O caso de uso deve retornar os dados produzidos pelo gerador."""
    fake_sky_map = make_skymap()

    generator = FakeSkyMapGenerator(
        result=fake_sky_map,
    )

    use_case = GenerateSkyMapUseCase(
        generator=generator,
    )

    input_dto = make_input_dto()

    result = use_case.execute(input_dto)

    assert isinstance(result, GenerateSkyMapOutputDTO)
    assert result.nside == 1
    assert result.npix == 12

    np.testing.assert_array_equal(
        result.map_data,
        fake_sky_map.data,
    )


def test_generate_skymap_delegates_correct_parameters() -> None:
    """O caso de uso deve encaminhar corretamente os dados ao gerador."""
    generator = FakeSkyMapGenerator(
        result=make_skymap(),
    )

    use_case = GenerateSkyMapUseCase(
        generator=generator,
    )

    input_dto = make_input_dto(
        nside=1,
        frequency_ghz=143.0,
        components=("cmb", "dust"),
    )

    use_case.execute(input_dto)

    assert len(generator.calls) == 1

    call = generator.calls[0]

    assert call["nside"] == NSide(1)
    assert call["freq_ghz"] == 143.0
    assert call["components"] == ("cmb", "dust")


# ============================================================
# execute() — validação do DTO
# ============================================================


def test_execute_rejects_none_input() -> None:
    """input_dto não pode ser None."""
    use_case = GenerateSkyMapUseCase(
        generator=FakeSkyMapGenerator(
            result=make_skymap(),
        )
    )

    with pytest.raises(
        ValueError,
        match="input_dto não pode ser None",
    ):
        use_case.execute(None)  # type: ignore[arg-type]


def test_execute_rejects_invalid_input_type() -> None:
    """input_dto deve ser GenerateSkyMapInputDTO."""
    use_case = GenerateSkyMapUseCase(
        generator=FakeSkyMapGenerator(
            result=make_skymap(),
        )
    )

    with pytest.raises(
        TypeError,
        match="input_dto deve ser GenerateSkyMapInputDTO",
    ):
        use_case.execute(object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "frequency_ghz",
    [0.0, -1.0, -100.0],
)
def test_execute_rejects_non_positive_frequency(
    frequency_ghz: float,
) -> None:
    """A frequência deve ser estritamente positiva."""
    generator = FakeSkyMapGenerator(
        result=make_skymap(),
    )

    use_case = GenerateSkyMapUseCase(
        generator=generator,
    )

    input_dto = make_input_dto(
        frequency_ghz=frequency_ghz,
    )

    with pytest.raises(
        ValueError,
        match="frequency_ghz deve ser positiva",
    ):
        use_case.execute(input_dto)

    assert generator.calls == []


@pytest.mark.parametrize(
    "nside",
    [0, -1, -8],
)
def test_execute_rejects_non_positive_nside(
    nside: int,
) -> None:
    """NSIDE deve ser estritamente positivo."""
    generator = FakeSkyMapGenerator(
        result=make_skymap(),
    )

    use_case = GenerateSkyMapUseCase(
        generator=generator,
    )

    input_dto = make_input_dto(
        nside=nside,
    )

    with pytest.raises(
        ValueError,
        match="nside deve ser positivo",
    ):
        use_case.execute(input_dto)

    assert generator.calls == []


def test_execute_rejects_empty_components() -> None:
    """Deve existir ao menos um componente físico."""
    generator = FakeSkyMapGenerator(
        result=make_skymap(),
    )

    use_case = GenerateSkyMapUseCase(
        generator=generator,
    )

    input_dto = make_input_dto(
        components=(),
    )

    with pytest.raises(
        ValueError,
        match="Deve haver ao menos um componente físico",
    ):
        use_case.execute(input_dto)

    assert generator.calls == []


@pytest.mark.parametrize(
    "components",
    [
        ("cmb", 123),
        (None,),
        (object(),),
    ],
)
def test_execute_rejects_non_string_components(
    components: tuple[object, ...],
) -> None:
    """Todos os componentes devem ser strings."""
    generator = FakeSkyMapGenerator(
        result=make_skymap(),
    )

    use_case = GenerateSkyMapUseCase(
        generator=generator,
    )

    input_dto = make_input_dto(
        components=components,
    )

    with pytest.raises(
        TypeError,
        match="Todos os componentes devem ser strings",
    ):
        use_case.execute(input_dto)

    assert generator.calls == []


# ============================================================
# execute() — contrato de retorno da porta
# ============================================================


@pytest.mark.parametrize(
    "invalid_result",
    [
        None,
        object(),
        np.ones(12, dtype=np.float64),
    ],
)
def test_execute_rejects_invalid_generator_result(
    invalid_result: object,
) -> None:
    """O gerador deve retornar uma entidade SkyMap."""
    generator = FakeSkyMapGenerator(
        result=invalid_result,
    )

    use_case = GenerateSkyMapUseCase(
        generator=generator,
    )

    with pytest.raises(
        TypeError,
        match="SkyMapGeneratorPort deve retornar SkyMap",
    ):
        use_case.execute(
            make_input_dto(),
        )

    assert len(generator.calls) == 1
