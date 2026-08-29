from __future__ import annotations

from typing import final

import numpy as np

from radioskysims_skymap.application.skymap.dto import (
    AnalyzeSkyMapInputDTO,
    AnalyzeSkyMapOutputDTO,
)
from radioskysims_skymap.application.skymap.ports import SkyMapAnalyzerPort
from radioskysims_skymap.domain.skymap import SkyMap


@final
class AnalyzeSkyMapUseCase:
    """
    Caso de uso responsável por orquestrar a análise espectral
    de mapas do céu.
    """

    def __init__(self, analyzer: SkyMapAnalyzerPort):
        if not isinstance(analyzer, SkyMapAnalyzerPort):
            raise TypeError(
                "analyzer deve implementar SkyMapAnalyzerPort"
            )

        self._analyzer = analyzer

    def execute(
        self,
        input_dto: AnalyzeSkyMapInputDTO,
    ) -> AnalyzeSkyMapOutputDTO:

        if input_dto is None:
            raise ValueError("input_dto não pode ser None")

        if not isinstance(input_dto, AnalyzeSkyMapInputDTO):
            raise TypeError(
                "input_dto deve ser AnalyzeSkyMapInputDTO"
            )

        sky_map = input_dto.sky_map
        lmax = input_dto.lmax

        if sky_map is None:
            raise ValueError("SkyMap não pode ser None")

        if not isinstance(sky_map, SkyMap):
            raise TypeError(
                "sky_map deve ser uma instância de SkyMap"
            )

        if lmax is not None and lmax <= 0:
            raise ValueError("lmax deve ser positivo")

        cls = self._analyzer.compute_cls(
            skymap=sky_map,
            lmax=lmax,
        )

        if not isinstance(cls, np.ndarray):
            raise TypeError(
                "SkyMapAnalyzerPort deve retornar numpy.ndarray"
            )

        return AnalyzeSkyMapOutputDTO(
            nside=sky_map.nside_value,
            lmax=lmax,
            cls=cls,
        )