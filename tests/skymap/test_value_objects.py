from __future__ import annotations

import healpy as hp
import numpy as np
import pytest

from radioskysims_skymap.domain.skymap.exceptions import InvalidNSideError
from radioskysims_skymap.domain.skymap.value_objects import NSide, SkyMapDataset

try:
    import xarray as xr
except Exception:
    xr = None

try:
    import dask.array as da
except Exception:
    da = None


# =============================================================================
# NSide
# =============================================================================


def test_nside_valido():
    n = NSide(16)

    assert n.value == 16


def test_nside_npix():
    n = NSide(16)

    assert n.npix == hp.nside2npix(16)


def test_nside_equality():
    assert NSide(16) == NSide(16)
    assert NSide(16) != NSide(32)


def test_nside_hash():
    assert hash(NSide(16)) == hash(NSide(16))


def test_nside_repr():
    assert repr(NSide(16)) == "NSide(16)"


@pytest.mark.parametrize(
    "value",
    [
        "16",
        1.5,
        None,
    ],
)
def test_nside_invalid_type(value):
    with pytest.raises(InvalidNSideError):
        NSide(value)


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        -8,
    ],
)
def test_nside_invalid_positive(value):
    with pytest.raises(InvalidNSideError):
        NSide(value)


def test_nside_above_limit():
    with pytest.raises(InvalidNSideError):
        NSide(16384)


# =============================================================================
# SkyMapDataset
# =============================================================================


def test_dataset_numpy():
    arr = np.arange(12)

    ds = SkyMapDataset(arr)

    assert ds.dataset is arr
    assert ds.shape == (12,)
    assert ds.ndim == 1
    assert ds.size == 12


def test_dataset_to_numpy_numpy():
    arr = np.arange(12)

    ds = SkyMapDataset(arr)

    out = ds.to_numpy()

    assert np.array_equal(out, arr)


def test_dataset_invalid():
    with pytest.raises(TypeError):
        SkyMapDataset("invalid dataset")


def test_dataset_list_converted():
    arr = np.arange(12)

    ds = SkyMapDataset(np.asarray(arr))

    assert np.array_equal(ds.to_numpy(), arr)


def test_infer_shape():
    arr = np.zeros((3, 4))

    ds = SkyMapDataset(arr)

    assert ds.shape == (3, 4)


def test_infer_ndim():
    arr = np.zeros((3, 4))

    ds = SkyMapDataset(arr)

    assert ds.ndim == 2


def test_infer_size():
    arr = np.zeros((3, 4))

    ds = SkyMapDataset(arr)

    assert ds.size == 12


# =============================================================================
# xarray (opcional)
# =============================================================================


@pytest.mark.skipif(xr is None, reason="xarray não instalado")
def test_xarray_dataset():
    arr = xr.DataArray(np.arange(12))

    ds = SkyMapDataset(arr)

    assert np.array_equal(ds.to_numpy(), np.arange(12))


# =============================================================================
# dask.array (opcional)
# =============================================================================


@pytest.mark.skipif(da is None, reason="dask não instalado")
def test_dask_array():
    arr = da.from_array(np.arange(12))

    ds = SkyMapDataset(arr)

    assert np.array_equal(ds.to_numpy(), np.arange(12))

# ---------------------------------------------------------------------
# SkyMapDataset — xarray com backend Dask
# ---------------------------------------------------------------------


def test_xarray_with_dask_backend_to_numpy() -> None:
    """
    DataArray com dados Dask deve ser computado e convertido
    para numpy.ndarray float64.
    """
    xr = pytest.importorskip("xarray")
    da = pytest.importorskip("dask.array")

    original = np.array(
        [1.0, 2.0, 3.0, 4.0],
        dtype=np.float32,
    )

    lazy_data = da.from_array(
        original,
        chunks=2,
    )

    data_array = xr.DataArray(
        lazy_data,
        dims=("pixel",),
    )

    dataset = SkyMapDataset(data_array)

    result = dataset.to_numpy()

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64

    assert dataset.dataset is data_array
    assert dataset.shape == (4,)
    assert dataset.ndim == 1
    assert dataset.size == 4

    np.testing.assert_array_equal(
        result,
        original.astype(np.float64),
    )

    # ---------------------------------------------------------------------
# SkyMapDataset — Dask Delayed
# ---------------------------------------------------------------------


def test_dask_delayed_to_numpy() -> None:
    """
    Um objeto Dask Delayed deve ser materializado como
    numpy.ndarray float64.
    """
    pytest.importorskip("dask")

    from dask import delayed

    @delayed
    def create_array() -> np.ndarray:
        return np.array(
            [1.0, 2.0, 3.0],
            dtype=np.float32,
        )

    lazy_data = create_array()

    dataset = SkyMapDataset(lazy_data)

    result = dataset.to_numpy()

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64

    assert dataset.dataset is lazy_data
    assert dataset.shape is None
    assert dataset.ndim is None
    assert dataset.size is None

    np.testing.assert_array_equal(
        result,
        np.array(
            [1.0, 2.0, 3.0],
            dtype=np.float64,
        ),
    )
