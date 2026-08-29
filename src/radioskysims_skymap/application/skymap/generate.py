# src/radioskysims/application/skymap/generate.py

from __future__ import annotations

from typing import final

from radioskysims_skymap.application.skymap.dto import (
    GenerateSkyMapInputDTO,
    GenerateSkyMapOutputDTO,
)
from radioskysims_skymap.application.skymap.ports import (
    ContinuumSurveyMapPort,
    SkyMapGeneratorPort,
)
from radioskysims_skymap.domain.skymap import NSide, SkyMap


@final
class GenerateSkyMapUseCase:
    """
    Caso de uso responsável pela geração de mapas do céu.

    A geração física é delegada à camada Infrastructure por meio da
    interface SkyMapGeneratorPort.
    """

    def __init__(self, generator: SkyMapGeneratorPort) -> None:
        if not isinstance(generator, SkyMapGeneratorPort):
            raise TypeError(
                "generator deve implementar SkyMapGeneratorPort"
            )

        self._generator = generator

    def execute(
        self,
        input_dto: GenerateSkyMapInputDTO,
    ) -> GenerateSkyMapOutputDTO:
        """
        Executa a geração de um SkyMap.
        """

        if input_dto is None:
            raise ValueError("input_dto não pode ser None")

        if not isinstance(input_dto, GenerateSkyMapInputDTO):
            raise TypeError(
                "input_dto deve ser GenerateSkyMapInputDTO"
            )

        if input_dto.frequency_ghz <= 0:
            raise ValueError(
                "frequency_ghz deve ser positiva"
            )

        if input_dto.nside <= 0:
            raise ValueError(
                "nside deve ser positivo"
            )

        if (
            input_dto.components is None
            or len(input_dto.components) == 0
        ):
            raise ValueError(
                "Deve haver ao menos um componente físico"
            )

        if not all(
            isinstance(component, str)
            for component in input_dto.components
        ):
            raise TypeError(
                "Todos os componentes devem ser strings"
            )

        nside = NSide(input_dto.nside)

        sky_map = self._generator.generate(
            nside=nside,
            freq_ghz=input_dto.frequency_ghz,
            components=input_dto.components,
        )

        if not isinstance(sky_map, SkyMap):
            raise TypeError(
                "SkyMapGeneratorPort deve retornar SkyMap"
            )

        return GenerateSkyMapOutputDTO(
            nside=sky_map.nside_value,
            npix=sky_map.npix,
            map_data=sky_map.data,
        )


@final
class ContinuumSurveyMapUseCase:
    """
    Caso de uso responsável por obter mapas contínuos
    provenientes de surveys astronômicos.
    """

    def __init__(
        self,
        provider: ContinuumSurveyMapPort,
    ) -> None:
        if not isinstance(provider, ContinuumSurveyMapPort):
            raise TypeError(
                "provider deve implementar ContinuumSurveyMapPort"
            )

        self._provider = provider

    def execute(
        self,
        *,
        survey_id: str,
        ra_deg: float,
        dec_deg: float,
        fov_deg: float,
        nside: int,
        freq_ghz: float | None = None,
        field: str = "I",
    ) -> SkyMap:
        """
        Obtém um SkyMap contínuo a partir de um survey.
        """

        if not survey_id.strip():
            raise ValueError(
                "survey_id não pode ser vazio"
            )

        if not (0.0 <= ra_deg < 360.0):
            raise ValueError(
                "ra_deg deve estar entre 0 e 360 graus"
            )

        if not (-90.0 <= dec_deg <= 90.0):
            raise ValueError(
                "dec_deg deve estar entre -90 e 90 graus"
            )

        if fov_deg <= 0:
            raise ValueError(
                "fov_deg deve ser positivo"
            )

        if nside <= 0:
            raise ValueError(
                "nside deve ser positivo"
            )

        if freq_ghz is not None and freq_ghz <= 0:
            raise ValueError(
                "freq_ghz deve ser positiva"
            )

        if field not in {"I", "Q", "U"}:
            raise ValueError(
                "field deve ser 'I', 'Q' ou 'U'"
            )

        nside_vo = NSide(nside)

        skymap = self._provider.fetch_cutout(
            survey_id=survey_id,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            fov_deg=fov_deg,
            nside=nside_vo,
            freq_ghz=freq_ghz,
            field=field,
        )

        if not isinstance(skymap, SkyMap):
            raise TypeError(
                "ContinuumSurveyMapPort deve retornar SkyMap"
            )

        return skymap