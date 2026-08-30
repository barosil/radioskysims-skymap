from __future__ import annotations

import matplotlib.pyplot as plt
import healpy as hp
import numpy as np

from radioskysims_skymap.domain.skymap.models import SkyMap
from radioskysims_skymap.infrastructure.skymap.telemetry import Telemetry

_telemetry = Telemetry(__name__)


class SkyMapVisualizer:
    """
    Visualização de mapas do céu e resultados derivados.

    Responsabilidade exclusiva:
    - Plotar mapas HEALPix
    - Plotar espectros angulares
    """

    # --------------------------------------------------
    # Mapas do céu
    # --------------------------------------------------
    def plot_map(
        self,
        skymap: SkyMap,
        *,
        title: str | None = None,
        unit: str | None = None,
        cmap: str = "viridis",
        min: float | None = None,
        max: float | None = None,
        coord: str | None = None,
        show: bool = True,
    ) -> None:
        """
        Plota um SkyMap em projeção Mollweide.
        """
        with _telemetry.measure("plot_map"):
            if not isinstance(skymap, SkyMap):
                msg = "skymap deve ser uma instância de SkyMap"
                raise TypeError(msg)

            hp.mollview(
                skymap.data,
                title=title or "Sky Map",
                unit=unit or skymap.unit,
                cmap=cmap,
                min=min,
                max=max,
                coord=coord,
            )

            _telemetry.event(
                "map_plotted",
                npix=len(skymap.data),
                unit=skymap.unit,
                cmap=cmap,
            )

            if show:
                plt.show()

    # --------------------------------------------------
    # Espectro angular
    # --------------------------------------------------
    def plot_cls(
        self,
        cls: np.ndarray,
        *,
        lmin: int = 0,
        lmax: int | None = None,
        title: str = "Angular Power Spectrum",
        show: bool = True,
    ) -> None:
        """
        Plota o espectro angular C_l.
        """
        with _telemetry.measure("plot_cls"):
            if not isinstance(cls, np.ndarray):
                msg = "cls deve ser um numpy.ndarray"
                raise TypeError(msg)

            ell = np.arange(len(cls))

            if lmax is not None:
                mask = (ell >= lmin) & (ell <= lmax)
                ell = ell[mask]
                cls = cls[mask]

            plt.figure()
            plt.loglog(ell, cls)
            plt.xlabel(r"$\ell$")
            plt.ylabel(r"$C_\ell$")
            plt.title(title)
            plt.grid(True, which="both", ls="--", alpha=0.5)

            _telemetry.event(
                "cls_plotted",
                cls_len=len(cls),
                lmin=lmin,
                lmax=lmax,
            )

            if show:
                plt.show()

    # --------------------------------------------------
    # Exportação
    # --------------------------------------------------
    def save_map(
        self,
        skymap: SkyMap,
        filepath: str,
        *,
        title: str | None = None,
        unit: str | None = None,
        cmap: str = "viridis",
    ) -> None:
        """
        Salva um mapa do céu como imagem.
        """
        with _telemetry.measure("save_map"):
            if not isinstance(skymap, SkyMap):
                msg = "skymap deve ser uma instância de SkyMap"
                raise TypeError(msg)

            hp.mollview(
                skymap.data,
                title=title or "Sky Map",
                unit=unit or skymap.unit,
                cmap=cmap,
            )

            plt.savefig(filepath, bbox_inches="tight")
            plt.close()

            _telemetry.event(
                "map_saved",
                filepath=filepath,
                npix=len(skymap.data),
                cmap=cmap,
            )
