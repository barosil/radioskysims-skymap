"""
Módulo: src/radioskysims/application/ports/skymap.py

Ports (Interfaces) da camada de Application para operacoes de SkyMap.

Define o contrato entre o domínio e as implementações concretas (HEALPy, PySM, etc).
Nenhuma lógica de negócio aqui - apenas abstrações.

Não importa bibliotecas concretas: healpy, pysm, numpy, etc.
Usa apenas abstrações do domínio: SkyMap, NSide.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from radioskysims_skymap.domain.skymap.models import SkyMap
from radioskysims_skymap.domain.skymap.value_objects import NSide


class SkyMapIOPort(ABC):
    """
    Port para operações de entrada/saída (I/O) de SkyMaps.

    Define o contrato para leitura e escrita de mapas do céu
    de/para formatos de arquivo ou outras fontes.
    """

    @abstractmethod
    def read_map(
        self,
        path: str | Path,
        *,
        nside: NSide | None = None,
        unit: str = "unknown",
        frame: str = "galactic",
        field: str = "I",
    ) -> SkyMap:
        """
        Lê um SkyMap de um arquivo.

        Args:
            path: Caminho do arquivo a ler.
            nside: Resolução HEALPix (opcional).
            unit: Unidade do mapa (padrão: "unknown").
            frame: Frame de referência (padrão: "galactic").
            field: Campo de Stokes (I, Q ou U; padrão: "I").

        Returns:
            SkyMap: Entidade de domínio com os dados lidos.

        Raises:
            FileNotFoundError: Se o arquivo não existir.
            ValueError: Se o arquivo não for válido.
        """
        raise NotImplementedError

    @abstractmethod
    def write_map(
        self,
        skymap: SkyMap,
        path: str | Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        """
        Escreve um SkyMap em arquivo.

        Args:
            skymap: Entidade SkyMap a escrever.
            path: Caminho de destino.
            overwrite: Se True, sobrescreve arquivo existente.

        Returns:
            Path: Caminho absoluto do arquivo escrito.

        Raises:
            FileExistsError: Se arquivo existe e overwrite=False.
            IOError: Erro ao escrever.
        """
        raise NotImplementedError


class SkyMapTransformPort(ABC):
    """
    Port para transformações geométricas e espectrais de SkyMaps.

    Define operações de processamento que modificam ou transformam
    a estrutura e/ou valores de um mapa do céu.
    """

    @abstractmethod
    def smooth(self, skymap: SkyMap, fwhm_rad: float) -> SkyMap:
        """
        Suaviza um SkyMap com um Gaussiano isotrópico.

        Args:
            skymap: Mapa a suavizar.
            fwhm_rad: Largura a meia altura do Gaussiano em radianos.

        Returns:
            SkyMap: Novo mapa suavizado.

        Raises:
            ValueError: Se fwhm_rad <= 0.
        """
        raise NotImplementedError

    @abstractmethod
    def rotate(
        self,
        skymap: SkyMap,
        rot: tuple[float, float, float],
    ) -> SkyMap:
        """
        Rotaciona um SkyMap usando ângulos de Euler.

        Args:
            skymap: Mapa a rotacionar.
            rot: Tupla (theta, phi, psi) em radianos.

        Returns:
            SkyMap: Novo mapa rotacionado.

        Raises:
            ValueError: Se ângulos forem inválidos.
        """
        raise NotImplementedError

    @abstractmethod
    def downgrade(self, skymap: SkyMap, target_nside: NSide) -> SkyMap:
        """
        Reduz a resolução (downgrade) de um SkyMap.

        Args:
            skymap: Mapa a fazer downgrade.
            target_nside: Resolução HEALPix alvo (menor que a atual).

        Returns:
            SkyMap: Novo mapa com resolução reduzida.

        Raises:
            ValueError: Se target_nside >= nside atual.
        """
        raise NotImplementedError
