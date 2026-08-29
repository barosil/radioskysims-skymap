from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from radioskysims_skymap.application.ports.skymap_repository import (
    SkyMapRepositoryPort,
)
from radioskysims_skymap.domain.skymap import NSide, SkyMap


class HDF5SkyMapRepository(SkyMapRepositoryPort):
    """
    Repository de SkyMap baseado em arquivos HDF5.

    Cada ``key`` corresponde a um arquivo::

        <root>/<key>.h5

    Estrutura HDF5
    ---------------
    /pixels
        Array HEALPix 1D.

    Attributes
    ----------
    nside
    npix
    unit
    frame
    field
    freq_ghz

    O repository preserva os mesmos metadados fundamentais
    utilizados pelo ``SkyMapRecord`` do backend SQL.
    """

    FILE_SUFFIX = ".h5"
    DATASET_NAME = "pixels"

    def __init__(
        self,
        root: str | Path,
    ) -> None:
        self._root = Path(root)

        self._root.mkdir(
            parents=True,
            exist_ok=True,
        )

    @property
    def root(self) -> Path:
        """Diretório raiz utilizado pelo repository."""
        return self._root

    def _validate_key(
        self,
        key: str,
    ) -> str:
        """
        Valida chave lógica antes de convertê-la em filename.

        Chaves não podem conter separadores de diretório para evitar
        que o repository escreva fora de ``root``.
        """

        if not isinstance(key, str):
            raise TypeError("SkyMap repository key must be a string.")

        normalized = key.strip()

        if not normalized:
            raise ValueError("SkyMap repository key cannot be empty.")

        if (
            "/" in normalized
            or "\\" in normalized
            or normalized in {".", ".."}
            or ".." in normalized
        ):
            raise ValueError("SkyMap repository key must not contain path separators or '..'.")

        return normalized

    def _path_for_key(
        self,
        key: str,
    ) -> Path:
        normalized = self._validate_key(key)

        return self._root / f"{normalized}{self.FILE_SUFFIX}"

    def save(
        self,
        key: str,
        skymap: SkyMap,
        *,
        overwrite: bool = False,
    ) -> str:
        """
        Persiste um SkyMap em HDF5.
        """

        if not isinstance(
            skymap,
            SkyMap,
        ):
            raise TypeError("skymap must be a SkyMap.")

        path = self._path_for_key(key)

        if path.exists() and not overwrite:
            raise FileExistsError(f"SkyMap already exists: {path}")

        data = np.asarray(skymap.data)

        if data.ndim != 1:
            raise ValueError("SkyMap HDF5 persistence requires a 1D HEALPix array.")

        # Grava em arquivo temporário e renomeia somente após
        # conclusão bem-sucedida. Isso reduz o risco de deixar
        # um arquivo final parcialmente escrito.
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")

        try:
            with h5py.File(
                temporary_path,
                "w",
            ) as handle:
                handle.create_dataset(
                    self.DATASET_NAME,
                    data=data,
                    compression="gzip",
                    shuffle=True,
                )

                handle.attrs["nside"] = int(skymap.nside_value)

                handle.attrs["npix"] = int(skymap.npix)

                handle.attrs["unit"] = str(skymap.unit)

                handle.attrs["frame"] = str(skymap.frame)

                handle.attrs["field"] = str(skymap.field)

                # HDF5 não representa None como atributo numérico.
                if skymap.freq_ghz is None:
                    handle.attrs["has_freq_ghz"] = False
                else:
                    handle.attrs["has_freq_ghz"] = True

                    handle.attrs["freq_ghz"] = float(skymap.freq_ghz)

            temporary_path.replace(path)

        except Exception:
            if temporary_path.exists():
                temporary_path.unlink()

            raise

        return str(path)

    def load(
        self,
        key: str,
    ) -> SkyMap | None:
        """
        Reconstrói um SkyMap a partir do arquivo HDF5.
        """

        path = self._path_for_key(key)

        if not path.exists():
            return None

        with h5py.File(
            path,
            "r",
        ) as handle:
            if self.DATASET_NAME not in handle:
                raise ValueError(
                    f"Invalid SkyMap HDF5 file: dataset {self.DATASET_NAME!r} not found."
                )

            data = np.asarray(handle[self.DATASET_NAME][...])

            required_attributes = {
                "nside",
                "unit",
                "frame",
                "field",
            }

            missing = required_attributes - set(handle.attrs.keys())

            if missing:
                raise ValueError(
                    f"Invalid SkyMap HDF5 file; missing attributes: {sorted(missing)}."
                )

            nside = int(handle.attrs["nside"])

            unit = str(handle.attrs["unit"])

            frame = str(handle.attrs["frame"])

            field = str(handle.attrs["field"])

            has_freq = bool(
                handle.attrs.get(
                    "has_freq_ghz",
                    "freq_ghz" in handle.attrs,
                )
            )

            freq_ghz: float | None

            if has_freq and "freq_ghz" in handle.attrs:
                freq_ghz = float(handle.attrs["freq_ghz"])

            else:
                freq_ghz = None

        return SkyMap(
            dataset=data,
            nside=NSide(nside),
            unit=unit,
            frame=frame,
            freq_ghz=freq_ghz,
            field=field,
        )

    def exists(
        self,
        key: str,
    ) -> bool:
        """Verifica se o mapa existe."""
        return self._path_for_key(key).is_file()

    def delete(
        self,
        key: str,
    ) -> bool:
        """
        Remove o arquivo associado à chave.
        """

        path = self._path_for_key(key)

        if not path.exists():
            return False

        path.unlink()

        return True
