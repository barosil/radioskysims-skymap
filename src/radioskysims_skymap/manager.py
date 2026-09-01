"""
Módulo: radioskysims_skymap/manager.py

Facade de conveniência (API pública unificada) para o bounded context
SkyMap, pensada para uso exploratório em notebooks e scripts.

``SkyMapManager`` concentra a criação/carregamento de mapas
(offline, PySM3, survey contínuo via HiPS) e retorna um
``SkyMapHandle``: um wrapper fino e encadeável em torno da entidade de
domínio ``SkyMap`` que expõe operações (suavização, rotação, máscara,
mudança de coordenadas), análise (estatísticas, espectro angular),
visualização e persistência (arquivo avulso ou registro por chave).

Este módulo não é um Use Case da camada Application: é um ponto de
composição de conveniência no nível do pacote (semelhante em espírito
ao Bootstrap/Composition Root do meta-pacote, porém escopado ao
pacote ``radioskysims_skymap``). Por isso, e apenas aqui, adapters
concretos de Infrastructure são instanciados diretamente. Fluxos
oficiais (CLI/GUI) devem continuar usando os Use Cases e Ports da
camada Application diretamente, não este módulo.

Copyright BINGO Collaboration
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import healpy as hp
import numpy as np
import numpy.typing as npt

from radioskysims_skymap.domain.skymap.analysis import SkyMapAnalysis
from radioskysims_skymap.domain.skymap.mask import SkyMask
from radioskysims_skymap.domain.skymap.models import SkyMap
from radioskysims_skymap.domain.skymap.operations import SkyMapOperations
from radioskysims_skymap.domain.skymap.value_objects import NSide
from radioskysims_skymap.infrastructure.skymap.providers.healpy import (
    HealpySkyMapProvider,
)
from radioskysims_skymap.infrastructure.skymap.providers.pysm import (
    PySMSkyMapGeneratorAdapter,
)
from radioskysims_skymap.infrastructure.skymap.repository.hdf5 import (
    HDF5SkyMapRepository,
)
from radioskysims_skymap.infrastructure.skymap.visualization import (
    SkyMapVisualizer,
)

_FRAME_TO_HP: Final[dict[str, str]] = {
    "galactic": "G",
    "equatorial": "C",
    "ecliptic": "E",
}

_DEFAULT_REGISTRY_DIR: Final[Path] = Path("data") / "skymap" / "registry"

FileFormat = Literal["fits", "hdf5"]
NoiseMethod = Literal["pysm", "gaussian"]


def _as_nside(nside: NSide | int) -> NSide:
    if isinstance(nside, NSide):
        return nside

    return NSide(nside)


@dataclass(frozen=True)
class SkyMapStatistics:
    """Resumo estatístico de um mapa (média/RMS/min/max)."""

    mean: float
    rms: float
    min: float
    max: float


class SkyMapHandle:
    """
    Wrapper fluente e imutável em torno de um ``SkyMap``.

    Cada operação (``smooth``, ``rotate``, ``mask``, ...) retorna um
    novo ``SkyMapHandle``, permitindo encadeamento::

        handle.smooth(fwhm_deg=5.0).rotate(target_frame="equatorial").view()
    """

    __slots__ = ("_skymap", "_visualizer", "_io_provider")

    def __init__(
        self,
        skymap: SkyMap,
        *,
        visualizer: SkyMapVisualizer | None = None,
        io_provider: HealpySkyMapProvider | None = None,
    ) -> None:
        self._skymap = skymap
        self._visualizer = visualizer or SkyMapVisualizer()
        self._io_provider = io_provider or HealpySkyMapProvider()

    # ------------------------------------------------------------------
    # Propriedades
    # ------------------------------------------------------------------

    @property
    def skymap(self) -> SkyMap:
        """Entidade de domínio encapsulada."""
        return self._skymap

    @property
    def data(self) -> npt.NDArray[np.float64]:
        return self._skymap.data

    @property
    def nside_value(self) -> int:
        return self._skymap.nside_value

    @property
    def unit(self) -> str:
        return self._skymap.unit

    @property
    def frame(self) -> str:
        return self._skymap.frame

    def __repr__(self) -> str:
        return f"SkyMapHandle({self._skymap!r})"

    # ------------------------------------------------------------------
    # Operações (encadeáveis)
    # ------------------------------------------------------------------

    def smooth(self, fwhm_deg: float) -> SkyMapHandle:
        """Aplica suavização Gaussiana (FWHM em graus)."""

        data = SkyMapOperations.smooth(
            self._skymap.data,
            fwhm_rad=np.radians(fwhm_deg),
        )
        return self._derive(data)

    def rotate(
        self,
        rot_deg: tuple[float, float, float] | None = None,
        *,
        target_frame: str | None = None,
    ) -> SkyMapHandle:
        """
        Rotaciona o mapa.

        Use ``rot_deg=(alpha, beta, gamma)`` para uma rotação livre em
        ângulos de Euler (graus), ou ``target_frame`` para converter
        entre sistemas de coordenadas (galactic/equatorial/ecliptic).
        """

        if target_frame is not None:
            return self.change_coordinates(target_frame)

        if rot_deg is None:
            msg = "Informe rot_deg ou target_frame."
            raise ValueError(msg)

        data = SkyMapOperations.rotate(
            self._skymap.data,
            tuple(np.radians(rot_deg)),
        )
        return self._derive(data)

    def change_coordinates(self, target_frame: str) -> SkyMapHandle:
        """Converte o mapa para outro sistema de coordenadas."""

        if target_frame not in _FRAME_TO_HP:
            msg = f"Frame inválido: {target_frame!r}."
            raise ValueError(msg)

        if target_frame == self._skymap.frame:
            return self

        rotator = hp.Rotator(
            coord=[
                _FRAME_TO_HP[self._skymap.frame],
                _FRAME_TO_HP[target_frame],
            ]
        )
        data = np.asarray(
            rotator.rotate_map_alms(self._skymap.data),
            dtype=np.float64,
        )
        return self._derive(data, frame=target_frame)

    def mask(self, mask_array: npt.NDArray[np.floating]) -> SkyMapHandle:
        """Aplica uma máscara (binária ou ponderada) ao mapa."""

        data = SkyMask.apply(self._skymap.data, mask_array)
        return self._derive(data)

    def downgrade(self, target_nside: NSide | int) -> SkyMapHandle:
        """Reduz a resolução HEALPix do mapa."""

        nside = _as_nside(target_nside)
        data = SkyMapOperations.downgrade(self._skymap.data, nside)

        new_map = SkyMap(
            dataset=data,
            nside=nside,
            unit=self._skymap.unit,
            frame=self._skymap.frame,
            freq_ghz=self._skymap.freq_ghz,
            field=self._skymap.field,
        )
        return SkyMapHandle(
            new_map,
            visualizer=self._visualizer,
            io_provider=self._io_provider,
        )

    # ------------------------------------------------------------------
    # Visualização
    # ------------------------------------------------------------------

    def view(self, **kwargs: object) -> SkyMapHandle:
        """Plota o mapa (Mollweide). Retorna ``self`` para encadeamento."""

        kwargs.setdefault("coord", _FRAME_TO_HP.get(self._skymap.frame))
        self._visualizer.plot_map(self._skymap, **kwargs)  # type: ignore[arg-type]
        return self

    # ------------------------------------------------------------------
    # Análise
    # ------------------------------------------------------------------

    def compute_statistics(self) -> SkyMapStatistics:
        """Calcula média, RMS, mínimo e máximo do mapa."""

        data = self._skymap.data
        lo, hi = SkyMapAnalysis.min_max(data)

        return SkyMapStatistics(
            mean=SkyMapAnalysis.mean(data),
            rms=SkyMapAnalysis.rms(data),
            min=lo,
            max=hi,
        )

    def compute_power_spectrum(
        self,
        lmax: int | None = None,
    ) -> npt.NDArray[np.float64]:
        """Calcula o espectro angular de potência Cℓ."""

        return SkyMapAnalysis.angular_power_spectrum(
            self._skymap.data,
            lmax=lmax,
        )

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------

    def save_to_file(
        self,
        path: str | Path,
        *,
        format: FileFormat = "fits",
        overwrite: bool = False,
    ) -> Path:
        """Salva o mapa em disco como FITS ou HDF5."""

        destination = Path(path)

        if format == "fits":
            return self._io_provider.write_map(
                self._skymap,
                destination,
                overwrite=overwrite,
            )

        if format == "hdf5":
            repository = HDF5SkyMapRepository(root=destination.parent)
            repository.save(
                destination.stem,
                self._skymap,
                overwrite=overwrite,
            )
            return destination

        msg = f"Formato não suportado: {format!r} (use 'fits' ou 'hdf5')."
        raise ValueError(msg)

    def register(
        self,
        key: str,
        *,
        root: str | Path | None = None,
        overwrite: bool = False,
    ) -> str:
        """Persiste o mapa no repositório HDF5 por chave lógica."""

        repository = HDF5SkyMapRepository(root=root or _DEFAULT_REGISTRY_DIR)
        return repository.save(key, self._skymap, overwrite=overwrite)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _derive(
        self,
        data: npt.NDArray[np.float64],
        *,
        frame: str | None = None,
    ) -> SkyMapHandle:
        new_map = SkyMap(
            dataset=data,
            nside=self._skymap.nside,
            unit=self._skymap.unit,
            frame=frame or self._skymap.frame,
            freq_ghz=self._skymap.freq_ghz,
            field=self._skymap.field,
        )
        return SkyMapHandle(
            new_map,
            visualizer=self._visualizer,
            io_provider=self._io_provider,
        )


class SkyMapManager:
    """
    Facade estática para criação e carregamento de ``SkyMapHandle``.

    Todos os métodos são ``classmethod`` e retornam um novo
    ``SkyMapHandle`` pronto para uso encadeado.
    """

    @classmethod
    def create_map_from_coefficients(
        cls,
        cls_array: npt.NDArray[np.floating],
        *,
        nside: NSide | int,
        lmax: int | None = None,
        seed: int | None = None,
        unit: str = "unknown",
        frame: str = "galactic",
        freq_ghz: float | None = None,
        field: str = "I",
    ) -> SkyMapHandle:
        """Gera um mapa sintético a partir de um espectro Cℓ (hp.synfast)."""

        if seed is not None:
            np.random.seed(seed)

        nside_vo = _as_nside(nside)
        map_data = np.asarray(
            hp.synfast(cls_array, nside=nside_vo.value, lmax=lmax),
            dtype=np.float64,
        )

        skymap = SkyMap(
            dataset=map_data,
            nside=nside_vo,
            unit=unit,
            frame=frame,
            freq_ghz=freq_ghz,
            field=field,
        )
        return SkyMapHandle(skymap)

    @classmethod
    def load_map_from_file(
        cls,
        path: str | Path,
        *,
        nside: NSide | int | None = None,
        unit: str = "unknown",
        frame: str = "galactic",
        field: str = "I",
    ) -> SkyMapHandle:
        """Carrega um mapa a partir de um arquivo FITS HEALPix."""

        provider = HealpySkyMapProvider()
        nside_vo = _as_nside(nside) if nside is not None else None
        skymap = provider.read_map(
            path,
            nside=nside_vo,
            unit=unit,
            frame=frame,
            field=field,
        )
        return SkyMapHandle(skymap, io_provider=provider)

    @classmethod
    def download(
        cls,
        *,
        survey_id: str,
        ra_deg: float,
        dec_deg: float,
        fov_deg: float,
        nside: NSide | int,
        freq_ghz: float | None = None,
        field: str = "I",
        base_path: str | Path | None = None,
    ) -> SkyMapHandle:
        """Obtém um recorte de um survey contínuo via HiPS/FITS local."""

        from radioskysims_skymap.infrastructure.skymap.providers.hips import (
            HiPSSkyMapProvider,
        )

        provider = HiPSSkyMapProvider(base_path=base_path)
        skymap = provider.fetch_cutout(
            survey_id=survey_id,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            fov_deg=fov_deg,
            nside=_as_nside(nside),
            freq_ghz=freq_ghz,
            field=field,
        )
        return SkyMapHandle(skymap)

    @classmethod
    def create_noise_map(
        cls,
        *,
        nside: NSide | int,
        freq_ghz: float | None = None,
        method: NoiseMethod = "pysm",
        components: list[str] | None = None,
        unit: str = "uK_CMB",
        frame: str = "galactic",
        field: str = "I",
        seed: int | None = None,
    ) -> SkyMapHandle:
        """Gera um mapa de "ruído"/componentes físicos.

        ``method="pysm"`` delega a componentes físicos do PySM3
        (synchrotron, dust, ...). ``method="gaussian"`` gera ruído
        Gaussiano branco por pixel, totalmente offline.
        """

        nside_vo = _as_nside(nside)

        if method == "pysm":
            if freq_ghz is None:
                msg = "freq_ghz é obrigatório para method='pysm'."
                raise ValueError(msg)

            generator = PySMSkyMapGeneratorAdapter(unit=unit, field=field)
            skymap = generator.generate(
                nside=nside_vo,
                freq_ghz=freq_ghz,
                components=components or ["synchrotron", "dust"],
            )
            return SkyMapHandle(skymap)

        if method == "gaussian":
            rng = np.random.default_rng(seed)
            data = rng.normal(size=nside_vo.npix)

            skymap = SkyMap(
                dataset=data,
                nside=nside_vo,
                unit=unit,
                frame=frame,
                freq_ghz=freq_ghz,
                field=field,
            )
            return SkyMapHandle(skymap)

        msg = f"Método de ruído não suportado: {method!r}."
        raise ValueError(msg)

    @classmethod
    def load_registered(
        cls,
        key: str,
        *,
        root: str | Path | None = None,
    ) -> SkyMapHandle | None:
        """Carrega um mapa previamente registrado via ``handle.register(...)``."""

        repository = HDF5SkyMapRepository(root=root or _DEFAULT_REGISTRY_DIR)
        skymap = repository.load(key)

        if skymap is None:
            return None

        return SkyMapHandle(skymap)


__all__ = [
    "SkyMapHandle",
    "SkyMapManager",
    "SkyMapStatistics",
]
