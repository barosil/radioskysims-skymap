"""Compat layer para provider HEALPy de SkyMap.

Este módulo preserva o caminho de import histórico
`radioskysims_skymap.infrastructure.skymap.healpy_provider.HealpySkyMapProvider`
apontando para a implementação canônica em
`radioskysims_skymap.infrastructure.skymap.providers.healpy`.
"""

from __future__ import annotations

from radioskysims_skymap.infrastructure.skymap.providers.healpy import HealpySkyMapProvider

__all__ = ["HealpySkyMapProvider"]
