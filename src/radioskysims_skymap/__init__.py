"""
Pacote: radioskysims_skymap

SkyMap do ecossistema RadioSkySim — domínio (SkyMap/NSide, mask, operations,
analysis), use cases de análise/geração, repositório HDF5, providers
(healpy/pysm/HIPS) e adapters.

Não contém `generate_continuum` (usa radiosources → pacote sources, Fase 4)
nem `tod_projector` (integração com TOD → pacote simulation, Fase 5).

Copyright BINGO Collaboration
"""

from __future__ import annotations

__version__ = "0.1.0"

from .domain.skymap import SkyMap
from .domain.skymap.value_objects import NSide
from .manager import SkyMapHandle, SkyMapManager, SkyMapStatistics

__all__ = [
    "NSide",
    "SkyMap",
    "SkyMapHandle",
    "SkyMapManager",
    "SkyMapStatistics",
]
