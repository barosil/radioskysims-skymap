"""
Exceções de domínio para o módulo SkyMap.

Centraliza exceções explícitas para operações e validações do SkyMap.
"""
from __future__ import annotations


class SkyMapError(Exception):
    """Exceção base para erros de domínio de SkyMap."""



class InvalidNSideError(SkyMapError):
    """Exceção levantada quando NSide é inválido."""



class DatasetNotFoundError(SkyMapError):
    """Exceção levantada quando dataset não é encontrado."""



class InvalidDatasetError(SkyMapError):
    """Exceção levantada quando dataset tem estrutura inválida."""



class IncompatibleShapeError(SkyMapError):
    """Exceção levantada quando shapes de operações são incompatíveis."""



class DownloadError(SkyMapError):
    """Exceção levantada durante operações de download de skymap."""



class CacheError(SkyMapError):
    """Exceção levantada durante operações de cache."""



class AnalysisError(SkyMapError):
    """Exceção levantada durante operações de análise de skymap."""

