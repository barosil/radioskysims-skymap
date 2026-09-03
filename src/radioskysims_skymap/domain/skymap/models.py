# src/radioskysims/domain/skymap/models.py

from __future__ import annotations

import healpy as hp
import numpy as np
import numpy.typing as npt

from radioskysims_skymap.domain.skymap.exceptions import InvalidDatasetError
from radioskysims_skymap.domain.skymap.value_objects import NSide, SkyMapDataset


class SkyMap:
    """
    Entidade de domínio que representa um mapa HEALPix.

    A entidade encapsula:

    - dados científicos;
    - resolução espacial (NSide);
    - metadados físicos.

    A entidade não realiza operações de entrada e saída.
    """

    __slots__ = (
        "_dataset",
        "_nside",
        "unit",
        "frame",
        "freq_ghz",
        "field",
    )

    def __init__(
        self,
        dataset: SkyMapDataset | npt.NDArray[np.floating],
        nside: NSide | int,
        unit: str,
        frame: str = "galactic",
        freq_ghz: float | None = None,
        field: str = "I",
    ) -> None:
        if not isinstance(nside, NSide):
            nside = NSide(nside)

        if not isinstance(dataset, SkyMapDataset):
            if not isinstance(dataset, np.ndarray):
                msg = "dataset must be a numpy.ndarray or SkyMapDataset"
                raise TypeError(msg)

            dataset = SkyMapDataset(np.asarray(dataset, dtype=np.float64))

        self._validate(
            data=dataset,
            nside=nside,
            unit=unit,
            frame=frame,
            field=field,
        )

        self._dataset = dataset
        self._nside = nside

        self.unit = unit
        self.frame = frame
        self.freq_ghz = freq_ghz
        self.field = field

    # ------------------------------------------------------------------
    # Validação
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(
        *,
        data: SkyMapDataset,
        nside: NSide,
        unit: str,
        frame: str,
        field: str,
    ) -> None:
        """
        Valida a consistência estrutural e física do mapa.
        """

        if data.ndim is not None and data.ndim != 1:
            msg = "SkyMap must be a 1D HEALPix map."
            raise InvalidDatasetError(msg)

        if data.size == 0:
            msg = "SkyMap dataset cannot be empty."
            raise InvalidDatasetError(msg)

        if data.size is not None:
            expected = hp.nside2npix(nside.value)

            if data.size != expected:
                msg = (
                    "Data size is inconsistent with NSide: "
                    f"expected {expected} pixels, received {data.size}."
                )
                raise InvalidDatasetError(msg)

        if not isinstance(unit, str):
            msg = "unit must be a string"
            raise TypeError(msg)

        if not unit.strip():
            msg = "Unit cannot be empty."
            raise InvalidDatasetError(msg)

        if not isinstance(frame, str):
            msg = "frame must be a string"
            raise TypeError(msg)

        if frame not in {
            "galactic",
            "equatorial",
            "ecliptic",
        }:
            msg = f"Invalid frame '{frame}'."
            raise InvalidDatasetError(msg)

        if not isinstance(field, str):
            msg = "field must be a string"
            raise TypeError(msg)

        if field not in {
            "I",
            "Q",
            "U",
        }:
            msg = "Field must be I, Q or U."
            raise InvalidDatasetError(msg)

    # ------------------------------------------------------------------
    # Propriedades
    # ------------------------------------------------------------------

    @property
    def dataset(self) -> SkyMapDataset:
        """
        Retorna o Value Object que encapsula o dataset.
        """
        return self._dataset

    @property
    def data(self) -> npt.NDArray[np.float64]:
        """
        Retorna uma cópia dos dados como ndarray float64.
        """
        return self._dataset.to_numpy().astype(
            np.float64,
            copy=True,
        )

    @property
    def nside(self) -> NSide:
        """
        Retorna o Value Object NSide.
        """
        return self._nside

    @property
    def nside_value(self) -> int:
        """
        Retorna o valor inteiro do NSide.
        """
        return self._nside.value

    @property
    def npix(self) -> int:
        """
        Retorna o número de pixels do mapa.
        """
        if self._dataset.size is not None:
            return self._dataset.size

        return self._nside.npix

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.npix

    def __repr__(self) -> str:
        return (
            "SkyMap("
            f"nside={self.nside_value}, "
            f"npix={self.npix}, "
            f"unit='{self.unit}', "
            f"frame='{self.frame}', "
            f"field='{self.field}'"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SkyMap):
            return False

        return (
            self.nside == other.nside
            and self.unit == other.unit
            and self.frame == other.frame
            and self.freq_ghz == other.freq_ghz
            and self.field == other.field
            and np.array_equal(
                self.data,
                other.data,
            )
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.nside_value,
                self.unit,
                self.frame,
                self.freq_ghz,
                self.field,
            )
        )

    # ------------------------------------------------------------------
    # Operações
    # ------------------------------------------------------------------

    def copy(self) -> SkyMap:
        """
        Retorna uma cópia profunda da entidade.
        """
        return SkyMap(
            dataset=self.data,
            nside=self.nside,
            unit=self.unit,
            frame=self.frame,
            freq_ghz=self.freq_ghz,
            field=self.field,
        )

    def apply_mask(
        self,
        mask: npt.NDArray[np.floating],
    ) -> SkyMap:
        """
        Retorna um novo mapa com a máscara aplicada.
        """

        if not isinstance(mask, np.ndarray):
            msg = "Mask must be numpy.ndarray."
            raise TypeError(msg)

        data = self.data

        if mask.shape != data.shape:
            msg = "Mask shape mismatch."
            raise ValueError(msg)

        masked_data = np.multiply(
            data,
            mask,
            dtype=np.float64,
        )

        return SkyMap(
            dataset=masked_data,
            nside=self.nside,
            unit=self.unit,
            frame=self.frame,
            freq_ghz=self.freq_ghz,
            field=self.field,
        )

    # ------------------------------------------------------------------
    # Healpy
    # ------------------------------------------------------------------

    def alm(
        self,
        lmax: int | None = None,
    ) -> npt.NDArray[np.complex128]:
        """
        Converte o mapa para coeficientes harmônicos esféricos.
        """
        result = hp.map2alm(
            self.data,
            lmax=lmax,
        )

        return np.asarray(
            result,
            dtype=np.complex128,
        )

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    @classmethod
    def from_array(
        cls,
        data: npt.NDArray[np.floating],
        nside: NSide,
        unit: str = "unknown",
        frame: str = "galactic",
        freq_ghz: float | None = None,
        field: str = "I",
    ) -> SkyMap:
        """
        Cria um SkyMap a partir de um ndarray.
        """
        return cls(
            dataset=data,
            nside=nside,
            unit=unit,
            frame=frame,
            freq_ghz=freq_ghz,
            field=field,
        )

    @classmethod
    def from_alm(
        cls,
        alm: npt.NDArray[np.complexfloating],
        nside: NSide,
        unit: str,
        frame: str = "galactic",
        freq_ghz: float | None = None,
        field: str = "I",
    ) -> SkyMap:
        """
        Cria um SkyMap a partir de coeficientes harmônicos esféricos.
        """
        data = hp.alm2map(
            alm,
            nside.value,
        )

        return cls(
            dataset=np.asarray(
                data,
                dtype=np.float64,
            ),
            nside=nside,
            unit=unit,
            frame=frame,
            freq_ghz=freq_ghz,
            field=field,
        )
