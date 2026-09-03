# src/radioskysims/domain/skymap/mask.py

"""
Domain Service for applying masks to HEALPix sky maps.
"""

from __future__ import annotations

from typing import final

import numpy as np
import numpy.typing as npt


@final
class SkyMask:
    """
    Domain Service para aplicação de máscaras em mapas HEALPix.

    Todas as operações são puras e não modificam
    o mapa original.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(
        map_data: npt.NDArray[np.floating],
        mask: npt.NDArray[np.floating],
    ) -> None:
        """
        Valida mapa e máscara.
        """

        if not isinstance(map_data, np.ndarray):
            msg = "map_data deve ser numpy.ndarray"
            raise TypeError(msg)

        if not isinstance(mask, np.ndarray):
            msg = "mask deve ser numpy.ndarray"
            raise TypeError(msg)

        if map_data.ndim != 1:
            msg = "map_data deve ser unidimensional"
            raise ValueError(msg)

        if mask.ndim != 1:
            msg = "mask deve ser unidimensional"
            raise ValueError(msg)

        if map_data.shape != mask.shape:
            msg = "map_data e mask devem possuir o mesmo shape"
            raise ValueError(msg)

        if map_data.size == 0:
            msg = "map_data não pode estar vazio"
            raise ValueError(msg)

        if mask.size == 0:
            msg = "mask não pode estar vazio"
            raise ValueError(msg)

        if not np.all(np.isfinite(map_data)):
            msg = "map_data contém valores não finitos"
            raise ValueError(msg)

        if not np.all(np.isfinite(mask)):
            msg = "mask contém valores não finitos"
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # Operações
    # ------------------------------------------------------------------

    @staticmethod
    def apply(
        map_data: npt.NDArray[np.floating],
        mask: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.float64]:
        """
        Aplica uma máscara a um mapa HEALPix.

        Parameters
        ----------
        map_data
            Mapa original.

        mask
            Máscara binária ou ponderada.

        Returns
        -------
        NDArray[np.float64]
            Novo mapa mascarado.
        """

        SkyMask._validate(
            map_data,
            mask,
        )

        return np.multiply(
            map_data,
            mask,
            dtype=np.float64,
        )


__all__ = [
    "SkyMask",
]
