# src/radioskysims/infrastructure/skymap/utils.py

from __future__ import annotations

import healpy as hp
import numpy as np

from radioskysims_skymap.domain.skymap import NSide


# --------------------------------------------------
# HEALPix helpers
# --------------------------------------------------
def expected_npix(nside: NSide | int) -> int:
    """
    Retorna o número esperado de pixels para um NSide.
    """
    value = nside.value if isinstance(nside, NSide) else int(nside)
    return int(hp.nside2npix(value))


def validate_healpix_map(data: np.ndarray, nside: NSide) -> None:
    """
    Valida se o array é compatível com HEALPix 1D.
    """
    if not isinstance(data, np.ndarray):
        msg = "Map data must be a numpy array"
        raise TypeError(msg)

    if data.ndim != 1:
        msg = "HEALPix map must be 1D"
        raise ValueError(msg)

    if data.size != expected_npix(nside):
        msg = "Data size does not match NSide"
        raise ValueError(msg)


# --------------------------------------------------
# Conversões
# --------------------------------------------------
def ensure_float_array(data: np.ndarray) -> np.ndarray:
    """
    Garante array float64 contínuo.
    """
    return np.asarray(data, dtype=float)


def mask_invalid(data: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """
    Substitui NaN / inf por valor definido.
    """
    cleaned = data.copy()
    cleaned[~np.isfinite(cleaned)] = fill_value
    return cleaned
