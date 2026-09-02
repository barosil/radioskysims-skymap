"""\
Módulo: infrastructure/skymap/io/planck.py

Downloader de mapas do Planck (FITS) com cache local.

Baixa um produto Planck (ex.: mapa SMICA) de uma URL configurável, com cache
em disco (``FileDownloader``), e oferece a carga como ``SkyMap`` (com
downgrade opcional para um NSide de trabalho).

> As URLs canônicas apontam para o arquivo da NASA LAMBDA. Dependendo da rede
> ou da reorganização do arquivo, elas podem precisar de ajuste — basta
> sobrescrever com ``url=...`` na chamada.

Copyright BINGO Collaboration
Last modified: 2026-09-01
"""

from __future__ import annotations

from pathlib import Path

from radioskysims_skymap.domain.skymap.exceptions import SkyMapError
from radioskysims_skymap.domain.skymap.value_objects import NSide
from radioskysims_skymap.infrastructure.skymap.io.download import FileDownloader

# URLs canônicas dos produtos Planck (NASA LAMBDA, arquivo Planck 2018).
PLANCK_PRODUCT_URLS: dict[str, str] = {
    "smica": (
        "https://lambda.gsfc.nasa.gov/data/planck/planck2018/smica/"
        "COM_CompMap_SMICA-raw-2048_R3.00_full.fits"
    ),
}

DEFAULT_PRODUCT = "smica"


class PlanckSkyMapDownloader:
    """Baixa (com cache) um mapa do Planck em FITS e o carrega como ``SkyMap``."""

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        downloader: FileDownloader | None = None,
    ) -> None:
        self._cache_dir = Path(cache_dir or "data/planck")
        self._downloader = downloader or FileDownloader(timeout=180)

    def download(
        self,
        product: str | None = None,
        *,
        url: str | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Baixa o FITS do produto (ou usa cache) e devolve o caminho local."""
        product = product or DEFAULT_PRODUCT
        base_url = url or PLANCK_PRODUCT_URLS.get(product)
        if base_url is None:
            msg = (
                f"produto Planck desconhecido: {product!r}; "
                f"disponíveis: {list(PLANCK_PRODUCT_URLS)}"
            )
            raise SkyMapError(msg)
        dest = self._cache_dir / Path(base_url).name
        if dest.exists() and not overwrite:
            print(f"[cache] Planck {product}: {dest}")
            return dest
        print(f"[online] baixando Planck {product} ...")
        return self._downloader.download(base_url, dest, overwrite=overwrite)

    def load(
        self,
        product: str | None = None,
        *,
        nside: int | None = None,
        frame: str = "galactic",
        field: str = "I",
        url: str | None = None,
        overwrite: bool = False,
    ):
        """Baixa e carrega o mapa como ``SkyMap`` (com downgrade opcional).

        Returns
        -------
        SkyMap
            Mapa em ``nside`` (se informado) ou na resolução nativa do FITS.
        """
        from radioskysims_skymap.infrastructure.skymap.providers.healpy import (
            HealpySkyMapProvider,
        )

        path = self.download(product, url=url, overwrite=overwrite)
        provider = HealpySkyMapProvider()
        if nside is None:
            return provider.read_map(path, unit="K_CMB", frame=frame, field=field)

        from radioskysims_skymap.domain.skymap.models import SkyMap
        from radioskysims_skymap.domain.skymap.operations import SkyMapOperations

        full = provider.read_map(path, unit="K_CMB", frame=frame, field=field)
        target = NSide(nside)
        data = SkyMapOperations.downgrade(full.data, target)
        return SkyMap(
            dataset=data,
            nside=target,
            unit="K_CMB",
            frame=frame,
            field=field,
        )
