"""Domínio de SkyMap - Value Objects, Entidades, Exceções e Serviços de Domínio."""

from __future__ import annotations

from .analysis import SkyMapAnalysis
from .exceptions import (
    AnalysisError,
    CacheError,
    DatasetNotFoundError,
    DownloadError,
    IncompatibleShapeError,
    InvalidDatasetError,
    InvalidNSideError,
    SkyMapError,
)
from .models import SkyMap
from .value_objects import NSide, SkyMapDataset

__all__ = [
    "AnalysisError",
    "CacheError",
    "DatasetNotFoundError",
    "DownloadError",
    "IncompatibleShapeError",
    "InvalidDatasetError",
    "InvalidNSideError",
    "NSide",
    "SkyMap",
    "SkyMapAnalysis",
    "SkyMapDataset",
    "SkyMapError",
]
