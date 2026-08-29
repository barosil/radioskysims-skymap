# src/radioskysims/application/skymap/ports.py

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from numpy.typing import NDArray
import numpy as np

from radioskysims_skymap.domain.skymap import NSide, SkyMap


class SkyMapGeneratorPort(ABC):
    """
    Porta para geração de mapas do céu.

    Implementações concretas pertencem à camada Infrastructure
    (PySM3, Healpy, WebSky, etc.).
    """

    @abstractmethod
    def generate(
        self,
        *,
        nside: NSide,
        freq_ghz: float,
        components: Sequence[str] | None = None,
    ) -> SkyMap:
        """
        Gera um mapa do céu para uma frequência e resolução.
        """
        raise NotImplementedError


class SkyMapAnalyzerPort(ABC):
    """
    Porta para análise científica de mapas do céu.

    Implementações normalmente utilizam Healpy, NaMaster,
    PolSpice ou outras bibliotecas especializadas.
    """

    @abstractmethod
    def compute_cls(
        self,
        *,
        skymap: SkyMap,
        lmax: int | None = None,
    ) -> NDArray[np.float64]:
        """
        Calcula o espectro angular de potência (Cℓ).

        Parameters
        ----------
        skymap
            Mapa do céu.
        lmax
            Multipolo máximo.

        Returns
        -------
        NDArray[np.float64]
            Espectro angular Cℓ.
        """
        raise NotImplementedError


class SkyMapMaskPort(ABC):
    """
    Porta para aplicação de máscaras em mapas do céu.
    """

    @abstractmethod
    def apply(
        self,
        *,
        skymap: SkyMap,
        mask: SkyMap,
    ) -> SkyMap:
        """
        Aplica uma máscara e retorna um novo SkyMap.
        """
        raise NotImplementedError


class ContinuumSurveyMapPort(ABC):
    """
    Porta para obtenção de mapas contínuos provenientes
    de surveys astronômicos (HiPS, FITS, Aladin etc.).
    """

    @abstractmethod
    def fetch_cutout(
        self,
        *,
        survey_id: str,
        ra_deg: float,
        dec_deg: float,
        fov_deg: float,
        nside: NSide,
        freq_ghz: float | None = None,
        field: str = "I",
    ) -> SkyMap:
        """
        Obtém um recorte (cutout) do survey.

        Parameters
        ----------
        survey_id
            Identificador do survey.
        ra_deg
            Ascensão reta em graus.
        dec_deg
            Declinação em graus.
        fov_deg
            Campo de visão em graus.
        nside
            Resolução HEALPix desejada.
        freq_ghz
            Frequência associada ao mapa.
        field
            Campo de Stokes (I, Q ou U).

        Returns
        -------
        SkyMap
            Mapa contínuo convertido para o modelo de domínio.
        """
        raise NotImplementedError