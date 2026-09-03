"""
Domain-level scientific analysis of HEALPix sky maps.

Pure mathematical operations without I/O.
"""

from __future__ import annotations

import importlib
from typing import Any, final

import numpy as np
import numpy.typing as npt


@final
class SkyMapAnalysis:
    """
    Domain Service para análises científicas de mapas HEALPix.

    Todas as operações são funções puras e independentes de I/O.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_map(
        map_data: npt.NDArray[np.floating],
    ) -> None:
        """
        Valida um mapa HEALPix.
        """

        if not isinstance(map_data, np.ndarray):
            msg = "map_data deve ser numpy.ndarray"
            raise TypeError(msg)

        if map_data.ndim != 1:
            msg = "map_data deve ser unidimensional"
            raise ValueError(msg)

        if map_data.size == 0:
            msg = "map_data não pode estar vazio"
            raise ValueError(msg)

        if not np.all(np.isfinite(map_data)):
            msg = "map_data contém valores não finitos"
            raise ValueError(msg)

    @staticmethod
    def _healpy() -> Any:
        """
        Importa healpy sob demanda.
        """

        try:
            return importlib.import_module("healpy")
        except ImportError as exc:
            msg = "healpy é necessário para calcular o espectro angular."
            raise ImportError(msg) from exc

    # ------------------------------------------------------------------
    # Estatísticas básicas
    # ------------------------------------------------------------------

    @staticmethod
    def mean(
        map_data: npt.NDArray[np.floating],
    ) -> float:
        """
        Calcula a média dos valores do mapa.
        """

        SkyMapAnalysis._validate_map(map_data)

        return float(np.mean(map_data))

    @staticmethod
    def rms(
        map_data: npt.NDArray[np.floating],
    ) -> float:
        """
        Calcula o Root Mean Square (RMS).
        """

        SkyMapAnalysis._validate_map(map_data)

        return float(np.sqrt(np.mean(np.square(map_data))))

    @staticmethod
    def min_max(
        map_data: npt.NDArray[np.floating],
    ) -> tuple[float, float]:
        """
        Retorna os valores mínimo e máximo.
        """

        SkyMapAnalysis._validate_map(map_data)

        return (
            float(np.min(map_data)),
            float(np.max(map_data)),
        )

    # ------------------------------------------------------------------
    # Espectro Angular
    # ------------------------------------------------------------------

    @staticmethod
    def angular_power_spectrum(
        map_data: npt.NDArray[np.floating],
        lmax: int | None = None,
    ) -> npt.NDArray[np.float64]:
        """
        Calcula o espectro angular Cℓ.

        Parameters
        ----------
        map_data
            Mapa HEALPix.

        lmax
            Multipolo máximo.

        Returns
        -------
        NDArray[np.float64]
            Espectro angular Cℓ.
        """

        SkyMapAnalysis._validate_map(map_data)

        if lmax is not None:
            if not isinstance(lmax, int):
                msg = "lmax deve ser inteiro"
                raise TypeError(msg)

            if lmax <= 0:
                msg = "lmax deve ser positivo"
                raise ValueError(msg)

        hp = SkyMapAnalysis._healpy()

        cls = hp.anafast(
            map_data,
            lmax=lmax,
        )

        return np.asarray(
            cls,
            dtype=np.float64,
        )


__all__ = [
    "SkyMapAnalysis",
]
