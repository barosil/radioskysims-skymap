"""\
Módulo: packages/radioskysims-skymap/src/radioskysims_skymap/application/orchestrators/skymap_orchestrator.py

Descrição:
    Fachada de alto nível (UX — Onda 3/R6) para mapas do céu (SkyMap).

    Encapsula o ``SkyMapManager``/``SkyMapHandle`` numa API encadeável que
    esconde o domínio/HEALPix:

        >>> skymap = SkyMapOrchestrator().from_noise(nside=64, method="gaussian")
        >>> skymap.smooth(fwhm_deg=5.0).view()
        >>> skymap.save("map.fits")

    Integrações ainda pendentes (documentadas): ``from_sources`` (mapa a
    partir de fontes — composição com radioskysims-sources no umbrella) e
    ``to_tod`` (projeção mapa→TOD — vive em radioskysims-simulation).

Copyright BINGO Collaboration
Last modified: 2026-09-03
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from radioskysims_skymap.manager import SkyMapHandle, SkyMapManager

__all__ = ["SkyMapOrchestrator"]


class SkyMapOrchestrator:
    """Fachada encadeável de mapas do céu (cria → opera → persiste → vê).

    Mantém um ``SkyMapHandle`` ativo; cada operação devolve ``self`` para
    encadeamento (builder). ``handle``/``data`` expõem o estado atual.
    """

    def __init__(
        self,
        handle: SkyMapHandle | None = None,
        *,
        manager: Any | None = None,
    ) -> None:
        self._handle = handle
        self._manager = manager or SkyMapManager

    # ──────────────────────────────────────────────────────────
    # Estado
    # ──────────────────────────────────────────────────────────
    @property
    def handle(self) -> SkyMapHandle | None:
        """Handle ativo (None antes da primeira criação/carga)."""
        return self._handle

    @property
    def data(self) -> npt.NDArray[np.float64]:
        """Dados do mapa ativo."""
        return self._require_handle().data

    @property
    def nside(self) -> int:
        """Resolução do mapa ativo."""
        return self._require_handle().nside_value

    @classmethod
    def from_handle(
        cls,
        handle: SkyMapHandle,
        *,
        manager: Any | None = None,
    ) -> "SkyMapOrchestrator":
        """Envolve um ``SkyMapHandle`` existente."""
        return cls(handle, manager=manager)

    # ──────────────────────────────────────────────────────────
    # Criação / carga
    # ──────────────────────────────────────────────────────────
    def from_noise(
        self,
        *,
        nside: int,
        freq_ghz: float | None = None,
        method: str = "pysm",
        components: list[str] | None = None,
        unit: str = "uK_CMB",
        frame: str = "galactic",
        field: str = "I",
        seed: int | None = None,
    ) -> "SkyMapOrchestrator":
        """Mapa de ruído/componentes (pysm ou gaussiano offline)."""
        self._handle = self._manager.create_noise_map(
            nside=nside,
            freq_ghz=freq_ghz,
            method=method,
            components=components,
            unit=unit,
            frame=frame,
            field=field,
            seed=seed,
        )
        return self

    def from_coefficients(
        self,
        cls_array: npt.NDArray[np.floating],
        *,
        nside: int,
        lmax: int | None = None,
        seed: int | None = None,
        unit: str = "unknown",
        frame: str = "galactic",
        freq_ghz: float | None = None,
        field: str = "I",
    ) -> "SkyMapOrchestrator":
        """Mapa sintético a partir de um espectro Cℓ (hp.synfast)."""
        self._handle = self._manager.create_map_from_coefficients(
            cls_array,
            nside=nside,
            lmax=lmax,
            seed=seed,
            unit=unit,
            frame=frame,
            freq_ghz=freq_ghz,
            field=field,
        )
        return self

    def load(
        self,
        path: str | Path,
        *,
        nside: int | None = None,
        unit: str = "unknown",
        frame: str = "galactic",
        field: str = "I",
    ) -> "SkyMapOrchestrator":
        """Carrega um mapa de um arquivo FITS HEALPix."""
        self._handle = self._manager.load_map_from_file(
            path,
            nside=nside,
            unit=unit,
            frame=frame,
            field=field,
        )
        return self

    def load_registered(
        self,
        key: str,
        *,
        root: str | Path | None = None,
    ) -> "SkyMapOrchestrator":
        """Carrega um mapa persistido no repositório HDF5 por chave lógica."""

        handle = self._manager.load_registered(
            key,
            root=root,
        )

        if handle is None:
            msg = f"Mapa registrado não encontrado: {key!r}."
            raise FileNotFoundError(msg)

        self._handle = handle
        return self

    def download(
        self,
        *,
        survey_id: str,
        ra_deg: float,
        dec_deg: float,
        fov_deg: float,
        nside: int,
        freq_ghz: float | None = None,
        field: str = "I",
        base_path: str | Path | None = None,
    ) -> "SkyMapOrchestrator":
        """Recorte de um survey contínuo (HiPS/FITS)."""
        self._handle = self._manager.download(
            survey_id=survey_id,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            fov_deg=fov_deg,
            nside=nside,
            freq_ghz=freq_ghz,
            field=field,
            base_path=base_path,
        )
        return self

    # ──────────────────────────────────────────────────────────
    # Operações (encadeáveis)
    # ──────────────────────────────────────────────────────────
    def apply(self, mask_array: npt.NDArray[np.floating]) -> "SkyMapOrchestrator":
        """Aplica uma máscara (binária/ponderada) ao mapa ativo."""
        self._handle = self._require_handle().mask(mask_array)
        return self

    def smooth(self, fwhm_deg: float) -> "SkyMapOrchestrator":
        """Suavização Gaussiana (FWHM em graus)."""
        self._handle = self._require_handle().smooth(fwhm_deg)
        return self

    def rotate(
        self,
        rot_deg: tuple[float, float, float] | None = None,
        *,
        target_frame: str | None = None,
    ) -> "SkyMapOrchestrator":
        """Rotaciona o mapa (Euler) ou converte de frame."""
        self._handle = self._require_handle().rotate(rot_deg, target_frame=target_frame)
        return self

    def downgrade(
        self,
        target_nside: int,
    ) -> "SkyMapOrchestrator":
        """Reduz a resolução HEALPix do mapa ativo."""

        self._handle = self._require_handle().downgrade(target_nside)
        return self

    # ──────────────────────────────────────────────────────────
    # Persistência / visualização / análise
    # ──────────────────────────────────────────────────────────
    def save(
        self,
        path: str | Path,
        *,
        format: str = "fits",
        overwrite: bool = False,
    ) -> Path:
        """Salva o mapa ativo em disco (FITS ou HDF5)."""
        return self._require_handle().save_to_file(
            path,
            format=format,  # type: ignore[arg-type]
            overwrite=overwrite,
        )

    def register(
        self,
        key: str,
        *,
        root: str | Path | None = None,
        overwrite: bool = False,
    ) -> str:
        """Persiste o mapa no repositório HDF5 por chave lógica."""
        return self._require_handle().register(key, root=root, overwrite=overwrite)

    def view(self, **kwargs: Any) -> "SkyMapOrchestrator":
        """Plota o mapa (Mollweide); retorna ``self`` para encadeamento."""
        self._require_handle().view(**kwargs)
        return self

    def statistics(self) -> Any:
        """Estatísticas (média/RMS/min/max) do mapa ativo."""
        return self._require_handle().compute_statistics()

    def power_spectrum(
        self,
        lmax: int | None = None,
    ) -> npt.NDArray[np.float64]:
        """Calcula o espectro angular de potência Cℓ do mapa ativo."""

        return self._require_handle().compute_power_spectrum(lmax=lmax)

    # ──────────────────────────────────────────────────────────
    # Integrações pendentes (documentadas)
    # ──────────────────────────────────────────────────────────
    def from_sources(self, *args: Any, **kwargs: Any) -> "SkyMapOrchestrator":
        """(Pendente) Gera o mapa a partir de fontes (radio/galaxy).

        Composição com ``radioskysims-sources`` (ContinuumSourceOrchestrator)
        a ser feita no umbrella — acompanhar o plano (Anexo A).
        """
        _ = (args, kwargs)
        msg = (
            "SkyMapOrchestrator.from_sources ainda não está disponível; a "
            "composição com fontes (radioskysims-sources) será feita no umbrella."
        )
        raise NotImplementedError(msg)

    def to_tod(self, *args: Any, **kwargs: Any) -> "SkyMapOrchestrator":
        """(Pendente) Projeta o mapa para TOD.

        A projeção mapa→TOD vive em ``radioskysims-simulation``
        (HealpySkyMapTODProjector).
        """
        _ = (args, kwargs)
        msg = (
            "SkyMapOrchestrator.to_tod ainda não está disponível; a projeção "
            "mapa→TOD vive em radioskysims-simulation."
        )
        raise NotImplementedError(msg)

    # ──────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────
    def _require_handle(self) -> SkyMapHandle:
        if self._handle is None:
            msg = (
                "Nenhum mapa ativo — chame from_noise/from_coefficients/load/"
                "download antes de operar."
            )
            raise ValueError(msg)
        return self._handle

    def __repr__(self) -> str:
        state = f"nside={self._handle.nside_value}" if self._handle else "vazio"
        return f"<SkyMapOrchestrator {state}>"
