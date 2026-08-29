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
            raise AnalysisError(
                "map_data deve ser numpy.ndarray"
            )

        if map_data.ndim != 1:
            raise AnalysisError(
                "map_data deve ser unidimensional"
            )

        if map_data.size == 0:
            raise AnalysisError(
                "map_data não pode estar vazio"
            )

        if not np.all(np.isfinite(map_data)):
            raise AnalysisError(
                "map_data contém valores não finitos"
            )

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
            raise AnalysisError(
                "fwhm_rad deve ser positivo"
            )

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
            raise AnalysisError(
                "rot deve conter (alpha, beta, gamma)"
            )

        if not all(
            isinstance(angle, (int, float))
            for angle in rot
        ):
            raise AnalysisError(
                "Todos os ângulos devem ser numéricos."
            )

        nside = hp.get_nside(map_data)

        alm = hp.map2alm(map_data)
        alm_rot = alm.copy()

        hp.rotate_alm(
            alm_rot,
            rot=rot,
        )

        return np.asarray(
            hp.alm2map(
                alm_rot,
                nside=nside,
                verbose=False,
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
            raise InvalidNSideError(
                "target_nside deve ser NSide"
            )

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