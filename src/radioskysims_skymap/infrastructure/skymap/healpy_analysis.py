"""Compat layer para analisadores HEALPy de SkyMap.

Preserva o caminho de import histórico
`radioskysims_skymap.infrastructure.skymap.healpy_analysis` apontando para
as implementações canônicas em `radioskysims_skymap.infrastructure.skymap.adapters.analysis`.
"""

from __future__ import annotations

from radioskysims_skymap.infrastructure.skymap.adapters.analysis import (
    HealpySkyMapAnalyzer,
    HealpySkyMapAnalyzerAdapter,
)

__all__ = ["HealpySkyMapAnalyzer", "HealpySkyMapAnalyzerAdapter"]
