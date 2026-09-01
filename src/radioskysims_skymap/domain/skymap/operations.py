from __future__ import annotations

from typing import final

import healpy as hp
import numpy as np
import numpy.typing as npt

from .exceptions import AnalysisError, InvalidNSideError
from .value_objects import NSide


@final
class SkyMapOperations:
    """
    Domain Service contendo operações físicas sobre mapas HEALPix.

    Todas as operações são funções puras, sem I/O.
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
            raise AnalysisError(msg)

        if map_data.ndim != 1:
            msg = "map_data deve ser unidimensional"
            raise AnalysisError(msg)

        if map_data.size == 0:
            msg = "map_data não pode estar vazio"
            raise AnalysisError(msg)

        if not np.all(np.isfinite(map_data)):
            msg = "map_data contém valores não finitos"
            raise AnalysisError(msg)

    # ------------------------------------------------------------------
    # Operações
    # ------------------------------------------------------------------

    @staticmethod
    def smooth(
        map_data: npt.NDArray[np.floating],
        fwhm_rad: float,
    ) -> npt.NDArray[np.float64]:
        """
        Aplica suavização Gaussiana.
        """

        SkyMapOperations._validate_map(map_data)

        if fwhm_rad <= 0:
            msg = "fwhm_rad deve ser positivo"
            raise AnalysisError(msg)

        return np.asarray(
            hp.smoothing(
                map_data,
                fwhm=fwhm_rad,
            ),
            dtype=np.float64,
        )

    @staticmethod
    def rotate(
        map_data: npt.NDArray[np.floating],
        rot: tuple[float, float, float],
    ) -> npt.NDArray[np.float64]:
        """
        Rotaciona um mapa usando ângulos de Euler.
        """

        SkyMapOperations._validate_map(map_data)

        if len(rot) != 3:
            msg = "rot deve conter (alpha, beta, gamma)"
            raise AnalysisError(msg)

        if not all(isinstance(angle, (int, float)) for angle in rot):
            msg = "Todos os ângulos devem ser numéricos."
            raise AnalysisError(msg)

        nside = hp.get_nside(map_data)

        alm = hp.map2alm(map_data)
        alm_rot = alm.copy()

        psi, theta, phi = rot

        hp.rotate_alm(
            alm_rot,
            psi=psi,
            theta=theta,
            phi=phi,
        )

        return np.asarray(
            hp.alm2map(
                alm_rot,
                nside=nside,
            ),
            dtype=np.float64,
        )

    @staticmethod
    def downgrade(
        map_data: npt.NDArray[np.floating],
        target_nside: NSide,
    ) -> npt.NDArray[np.float64]:
        """
        Reduz a resolução de um mapa HEALPix.
        """

        SkyMapOperations._validate_map(map_data)

        if not isinstance(target_nside, NSide):
            msg = "target_nside deve ser NSide"
            raise InvalidNSideError(msg)

        downgraded = hp.ud_grade(
            map_data,
            nside_out=target_nside.value,
            order_in="RING",
            order_out="RING",
            power=-2,
        )

        return np.asarray(
            downgraded,
            dtype=np.float64,
        )


__all__ = [
    "SkyMapOperations",
]
