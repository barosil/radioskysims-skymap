from __future__ import annotations

from typing import Any, TypeAlias

import healpy as hp
import numpy as np
import numpy.typing as npt

from .exceptions import InvalidNSideError

try:  # optional dependency
    import xarray as xr
except Exception:  # pragma: no cover
    xr = None

try:  # optional dependency
    import dask.array as da
    from dask.delayed import Delayed
except Exception:  # pragma: no cover
    da = None
    Delayed = None


# =============================================================================
# Aliases de tipagem
# =============================================================================

ArrayLike: TypeAlias = npt.NDArray[np.float64]


# =============================================================================
# NSide
# =============================================================================


class NSide:
    """
    Value Object que representa um NSide válido do HEALPix.
    """

    __slots__ = ("_value",)

    def __init__(self, value: int):
        self._validate(value)
        self._value = value

    @staticmethod
    def _validate(value: int) -> None:
        if not isinstance(value, int):
            msg = "NSide deve ser um inteiro"
            raise InvalidNSideError(msg)

        if value < 1:
            msg = "NSide deve ser >= 1"
            raise InvalidNSideError(msg)

        if not hp.isnsideok(value, nest=True):
            msg = "NSide deve ser potência de 2 (healpy)"
            raise InvalidNSideError(msg)

        if value > 8192:
            msg = "NSide acima do limite prático (8192)"
            raise InvalidNSideError(msg)

    @property
    def value(self) -> int:
        return self._value

    @property
    def npix(self) -> int:
        return hp.nside2npix(self._value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NSide) and self._value == other._value

    def __hash__(self) -> int:
        return hash(self._value)

    def __repr__(self) -> str:
        return f"NSide({self._value})"


# =============================================================================
# SkyMapDataset
# =============================================================================


class SkyMapDataset:
    """
    Value Object para encapsular datasets HEALPix.

    Tipos suportados
    ----------------
    - numpy.ndarray
    - xarray.DataArray
    - dask.array.Array
    - dask.delayed.Delayed
    """

    __slots__ = (
        "_dataset",
        "_shape",
        "_ndim",
        "_size",
    )

    def __init__(self, dataset: Any):
        self._validate(dataset)

        self._dataset = dataset
        self._shape = self._infer_shape(dataset)
        self._ndim = self._infer_ndim(dataset)
        self._size = self._infer_size(dataset)

    @property
    def dataset(self) -> Any:
        return self._dataset

    @property
    def shape(self) -> tuple[int, ...] | None:
        return self._shape

    @property
    def ndim(self) -> int | None:
        return self._ndim

    @property
    def size(self) -> int | None:
        return self._size

    def to_numpy(self) -> ArrayLike:
        """
        Materializa o dataset como numpy.ndarray.
        """

        data = self._dataset

        if isinstance(data, np.ndarray):
            return np.asarray(data, dtype=np.float64)

        if xr is not None and isinstance(data, xr.DataArray):
            values = data.data

            if da is not None and isinstance(values, da.Array):
                return np.asarray(
                    values.compute(),
                    dtype=np.float64,
                )

            return np.asarray(
                values,
                dtype=np.float64,
            )

        if da is not None and isinstance(data, da.Array):
            return np.asarray(
                data.compute(),
                dtype=np.float64,
            )

        if Delayed is not None and isinstance(data, Delayed):
            return np.asarray(
                data.compute(),
                dtype=np.float64,
            )

        return np.asarray(
            data,
            dtype=np.float64,
        )

    @staticmethod
    def _infer_shape(
        dataset: Any,
    ) -> tuple[int, ...] | None:
        if Delayed is not None and isinstance(dataset, Delayed):
            return None

        shape = getattr(dataset, "shape", None)

        if isinstance(shape, tuple):
            return tuple(int(dimension) for dimension in shape)

        return None

    @staticmethod
    def _infer_ndim(
        dataset: Any,
    ) -> int | None:
        if Delayed is not None and isinstance(dataset, Delayed):
            return None

        ndim = getattr(dataset, "ndim", None)

        if isinstance(ndim, (int, np.integer)):
            return int(ndim)

        return None

    @staticmethod
    def _infer_size(
        dataset: Any,
    ) -> int | None:
        if Delayed is not None and isinstance(dataset, Delayed):
            return None

        size = getattr(dataset, "size", None)

        if isinstance(size, (int, np.integer)):
            return int(size)

        return None

    @staticmethod
    def _validate(dataset: Any) -> None:
        is_numpy = isinstance(dataset, np.ndarray)

        is_xarray = xr is not None and isinstance(dataset, xr.DataArray)

        is_dask_array = da is not None and isinstance(dataset, da.Array)

        is_dask_delayed = Delayed is not None and isinstance(dataset, Delayed)

        if not (is_numpy or is_xarray or is_dask_array or is_dask_delayed):
            msg = (
                "Dataset deve ser numpy.ndarray, "
                "xarray.DataArray, "
                "dask.array.Array ou "
                "dask.delayed.Delayed."
            )
            raise TypeError(msg)
