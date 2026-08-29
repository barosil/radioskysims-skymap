# src/radioskysims/application/skymap/dto.py

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from radioskysims_skymap.domain.skymap import SkyMap


# ---------------------------------------------------------------------
# Tipos auxiliares
# ---------------------------------------------------------------------

SkyComponent: TypeAlias = Literal[
    "cmb",
    "synchrotron",
    "dust",
    "freefree",
    "ame",
]


# ---------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class GenerateSkyMapInputDTO:
    """
    Dados de entrada para geração de um SkyMap.
    """

    frequency_ghz: float
    nside: int
    components: Sequence[SkyComponent]


@dataclass(slots=True, frozen=True)
class GenerateSkyMapOutputDTO:
    """
    Resultado da geração de um SkyMap.
    """

    nside: int
    npix: int
    map_data: NDArray[np.float64]


# ---------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class AnalyzeSkyMapInputDTO:
    """
    Dados de entrada para análise de um SkyMap.
    """

    sky_map: SkyMap
    lmax: int | None = None


@dataclass(slots=True, frozen=True)
class AnalyzeSkyMapOutputDTO:
    """
    Resultado da análise espectral de um SkyMap.
    """

    nside: int
    lmax: int | None
    cls: NDArray[np.float64]