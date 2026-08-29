from __future__ import annotations

import healpy as hp
import numpy as np

from radioskysims_skymap.application.skymap.ports import SkyMapAnalyzerPort
from radioskysims_skymap.domain.skymap.models import SkyMap
from radioskysims_skymap.infrastructure.skymap.telemetry import Telemetry

_telemetry = Telemetry(__name__)


class HealpySkyMapAnalyzer(SkyMapAnalyzerPort):
    """
    Adapter de análise espectral usando HEALPy.
        """
    def compute_cls(
        self,
        skymap: SkyMap,
        lmax: int | None = None,
    ) -> np.ndarray:
        with _telemetry.measure("compute_cls"):
            data = skymap.data

            if data.ndim != 1:
                msg = "SkyMap data must be a 1D HEALPix map"
                raise ValueError(msg)

            result = np.asarray(hp.anafast(data, lmax=lmax))
            _telemetry.event(
                "cls_computed",
                npix=len(data),
                lmax=lmax,
                cls_len=len(result),
            )
            return result


class HealpySkyMapAnalyzerAdapter:
    """
    Adapter simplificado para análises HEALPy a partir de arrays.
    """

    def compute_cls(
        self,
        *,
        map_data: np.ndarray,
        lmax: int | None = None,
    ) -> np.ndarray:
        with _telemetry.measure("compute_cls_raw"):
            if not isinstance(map_data, np.ndarray):
                msg = "map_data must be a numpy.ndarray"
                raise TypeError(msg)

            if map_data.ndim != 1:
                msg = "map_data must be a 1D HEALPix map"
                raise ValueError(msg)

            result = np.asarray(hp.anafast(map_data, lmax=lmax))
            _telemetry.event(
                "cls_raw_computed",
                npix=len(map_data),
                lmax=lmax,
                cls_len=len(result),
            )
            return result
