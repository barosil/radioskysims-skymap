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
            msg = "analyzer deve implementar SkyMapAnalyzerPort"
            raise TypeError(msg)

        self._analyzer = analyzer

    def execute(
        self,
        input_dto: AnalyzeSkyMapInputDTO,
    ) -> AnalyzeSkyMapOutputDTO:
        if input_dto is None:
            msg = "input_dto não pode ser None"
            raise ValueError(msg)

        if not isinstance(input_dto, AnalyzeSkyMapInputDTO):
            msg = "input_dto deve ser AnalyzeSkyMapInputDTO"
            raise TypeError(msg)

        sky_map = input_dto.sky_map
        lmax = input_dto.lmax

        if sky_map is None:
            msg = "SkyMap não pode ser None"
            raise ValueError(msg)

        if not isinstance(sky_map, SkyMap):
            msg = "sky_map deve ser uma instância de SkyMap"
            raise TypeError(msg)

        if lmax is not None and lmax <= 0:
            msg = "lmax deve ser positivo"
            raise ValueError(msg)

        cls = self._analyzer.compute_cls(
            skymap=sky_map,
            lmax=lmax,
        )

        if not isinstance(cls, np.ndarray):
            msg = "SkyMapAnalyzerPort deve retornar numpy.ndarray"
            raise TypeError(msg)

        return AnalyzeSkyMapOutputDTO(
            nside=sky_map.nside_value,
            lmax=lmax,
            cls=cls,
        )
