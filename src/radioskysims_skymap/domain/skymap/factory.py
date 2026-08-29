from __future__ import annotations

import numpy as np
import numpy.typing as npt

from radioskysims_skymap.domain.skymap.models import SkyMap
from radioskysims_skymap.domain.skymap.value_objects import NSide, SkyMapDataset
from radioskysims_core.logging_config import get_logger

logger = get_logger("radioskysims.skymap")


class SkyMapFactory:
    """
    Factory responsável por construir entidades SkyMap.

    Centraliza metadados padrão e garante a criação consistente
    das entidades do domínio.

    Esta classe não realiza I/O nem depende de bibliotecas
    científicas externas como PySM3.
    """

    __slots__ = (
        "_unit",
        "_frame",
        "_field",
    )

    def __init__(
        self,
        *,
        unit: str = "unknown",
        frame: str = "galactic",
        field: str = "I",
    ) -> None:
        self._unit = unit
        self._frame = frame
        self._field = field

    def create(
        self,
        *,
        dataset: SkyMapDataset | npt.NDArray[np.floating],
        nside: NSide | int,
        freq_ghz: float | None = None,
        unit: str | None = None,
        frame: str | None = None,
        field: str | None = None,
    ) -> SkyMap:
        """
        Cria uma entidade SkyMap.

        Parameters
        ----------
        dataset
            Dados do mapa.

        nside
            Resolução HEALPix. Aceita NSide ou inteiro válido.

        freq_ghz
            Frequência de observação.

        unit
            Unidade física.

        frame
            Sistema de coordenadas.

        field
            Campo de Stokes.

        Returns
        -------
        SkyMap
            Nova entidade de domínio.
        """

        if isinstance(nside, bool):
            raise TypeError(
                "nside deve ser NSide ou int (bool não é permitido)."
            )

        if isinstance(nside, int):
            nside = NSide(nside)
        elif not isinstance(nside, NSide):
            raise TypeError(
                "nside deve ser uma instância de NSide ou int."
            )

        if freq_ghz is not None and freq_ghz <= 0:
            raise ValueError(
                "freq_ghz deve ser positiva."
            )

        logger.info(
            "skymap_create_start",
            nside=nside.value,
            freq_ghz=freq_ghz,
            frame=frame or self._frame,
            field=field or self._field,
        )

        try:
            skymap = SkyMap(
                dataset=dataset,
                nside=nside,
                unit=unit or self._unit,
                frame=frame or self._frame,
                freq_ghz=freq_ghz,
                field=field or self._field,
            )

            logger.info(
                "skymap_create_success",
                nside=skymap.nside_value,
                npix=skymap.npix,
                unit=skymap.unit,
                frame=skymap.frame,
                field=skymap.field,
            )

            return skymap

        except Exception:
            logger.exception(
                "skymap_create_failed",
                nside=nside.value,
                freq_ghz=freq_ghz,
            )
            raise