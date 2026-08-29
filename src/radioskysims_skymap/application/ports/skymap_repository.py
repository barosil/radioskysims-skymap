from __future__ import annotations

from typing import Protocol

from radioskysims_skymap.domain.skymap import SkyMap


class SkyMapRepositoryPort(Protocol):
    """
    Port para persistência de mapas HEALPix.

    O contrato não define tecnologia de armazenamento.
    Implementações concretas podem usar HDF5, banco de dados,
    armazenamento remoto etc.
    """

    def save(
        self,
        key: str,
        skymap: SkyMap,
        *,
        overwrite: bool = False,
    ) -> str:
        """
        Persiste um SkyMap.

        Parameters
        ----------
        key
            Identificador lógico do mapa.

        skymap
            Mapa HEALPix a persistir.

        overwrite
            Se True, permite substituir um mapa existente.

        Returns
        -------
        str
            Caminho ou identificador físico da persistência.
        """
        ...

    def load(
        self,
        key: str,
    ) -> SkyMap | None:
        """
        Carrega um SkyMap.

        Retorna None quando a chave não existir.
        """
        ...

    def exists(
        self,
        key: str,
    ) -> bool:
        """Retorna True quando a chave estiver persistida."""
        ...

    def delete(
        self,
        key: str,
    ) -> bool:
        """
        Remove um mapa persistido.

        Retorna True quando um arquivo foi removido.
        """
        ...
