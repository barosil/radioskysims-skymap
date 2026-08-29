# src/radioskysims/infrastructure/skymap/visualization.py

from __future__ import annotations

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from radioskysims_skymap.domain.skymap import SkyMap


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
        if not isinstance(skymap, SkyMap):
            raise TypeError("skymap deve ser uma instância de SkyMap")

        hp.mollview(
            skymap.data,
            title=title or "Sky Map",
            unit=unit or skymap.unit,
            cmap=cmap,
            min=min,
            max=max,
            coord=coord,
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
        if not isinstance(cls, np.ndarray):
            raise TypeError("cls deve ser um numpy.ndarray")

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
        if not isinstance(skymap, SkyMap):
            raise TypeError("skymap deve ser uma instância de SkyMap")

        hp.mollview(
            skymap.data,
            title=title or "Sky Map",
            unit=unit or skymap.unit,
            cmap=cmap,
        )

        plt.savefig(filepath, bbox_inches="tight")
        plt.close()
