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
            msg = "generator deve implementar SkyMapGeneratorPort"
            raise TypeError(msg)

        self._generator = generator

    def execute(
        self,
        input_dto: GenerateSkyMapInputDTO,
    ) -> GenerateSkyMapOutputDTO:
        """
        Executa a geração de um SkyMap.
        """

        if input_dto is None:
            msg = "input_dto não pode ser None"
            raise ValueError(msg)

        if not isinstance(input_dto, GenerateSkyMapInputDTO):
            msg = "input_dto deve ser GenerateSkyMapInputDTO"
            raise TypeError(msg)

        if input_dto.frequency_ghz <= 0:
            msg = "frequency_ghz deve ser positiva"
            raise ValueError(msg)

        if input_dto.nside <= 0:
            msg = "nside deve ser positivo"
            raise ValueError(msg)

        if input_dto.components is None or len(input_dto.components) == 0:
            msg = "Deve haver ao menos um componente físico"
            raise ValueError(msg)

        if not all(isinstance(component, str) for component in input_dto.components):
            msg = "Todos os componentes devem ser strings"
            raise TypeError(msg)

        nside = NSide(input_dto.nside)

        sky_map = self._generator.generate(
            nside=nside,
            freq_ghz=input_dto.frequency_ghz,
            components=input_dto.components,
        )

        if not isinstance(sky_map, SkyMap):
            msg = "SkyMapGeneratorPort deve retornar SkyMap"
            raise TypeError(msg)

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
            msg = "provider deve implementar ContinuumSurveyMapPort"
            raise TypeError(msg)

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
            msg = "survey_id não pode ser vazio"
            raise ValueError(msg)

        if not (0.0 <= ra_deg < 360.0):
            msg = "ra_deg deve estar entre 0 e 360 graus"
            raise ValueError(msg)

        if not (-90.0 <= dec_deg <= 90.0):
            msg = "dec_deg deve estar entre -90 e 90 graus"
            raise ValueError(msg)

        if fov_deg <= 0:
            msg = "fov_deg deve ser positivo"
            raise ValueError(msg)

        if nside <= 0:
            msg = "nside deve ser positivo"
            raise ValueError(msg)

        if freq_ghz is not None and freq_ghz <= 0:
            msg = "freq_ghz deve ser positiva"
            raise ValueError(msg)

        if field not in {"I", "Q", "U"}:
            msg = "field deve ser 'I', 'Q' ou 'U'"
            raise ValueError(msg)

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
            msg = "ContinuumSurveyMapPort deve retornar SkyMap"
            raise TypeError(msg)

        return skymap
