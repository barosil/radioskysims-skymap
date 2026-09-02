"""\
Módulo: tests/skymap/test_planck.py

Descrição:
    Testes do ``PlanckSkyMapDownloader`` (cache + carga/downgrade).

Copyright BINGO Collaboration
Last modified: 2026-09-01
"""

from __future__ import annotations

import healpy as hp
import numpy as np
import pytest

from radioskysims_skymap.domain.skymap import NSide, SkyMap
from radioskysims_skymap.domain.skymap.exceptions import SkyMapError
from radioskysims_skymap.infrastructure.skymap.io.planck import (
    DEFAULT_PRODUCT,
    PlanckSkyMapDownloader,
)
from radioskysims_skymap.infrastructure.skymap.providers.healpy import (
    HealpySkyMapProvider,
)


def _default_dest_name() -> str:
    from radioskysims_skymap.infrastructure.skymap.io.planck import (
        PLANCK_PRODUCT_URLS,
    )

    import urllib.parse

    return urllib.parse.unquote(PLANCK_PRODUCT_URLS[DEFAULT_PRODUCT].rsplit("/", 1)[-1])


def _write_fake_planck(cache_dir, nside: int = 32) -> None:
    """Grava um FITS HEALPix com o nome do produto default (simula o Planck)."""
    data = np.random.default_rng(0).normal(size=hp.nside2npix(nside))
    path = cache_dir / _default_dest_name()
    HealpySkyMapProvider().write_map(
        SkyMap(dataset=data, nside=nside, unit="K_CMB"), path, overwrite=True
    )
    return path


def test_download_uses_cache(tmp_path):
    cache = tmp_path / "planck"
    cache.mkdir(parents=True, exist_ok=True)
    fake = _write_fake_planck(cache)
    dl = PlanckSkyMapDownloader(cache_dir=cache)
    assert dl.download() == fake  # cache hit, sem rede


def test_load_downgrade(tmp_path):
    cache = tmp_path / "planck"
    cache.mkdir(parents=True, exist_ok=True)
    _write_fake_planck(cache, nside=32)
    dl = PlanckSkyMapDownloader(cache_dir=cache)
    sm = dl.load(nside=16)
    assert sm.nside_value == 16
    assert sm.npix == NSide(16).npix
    assert np.isfinite(sm.data).all()


def test_unknown_product_raises(tmp_path):
    dl = PlanckSkyMapDownloader(cache_dir=tmp_path)
    with pytest.raises(SkyMapError, match="produto Planck desconhecido"):
        dl.download("nao_existe")
