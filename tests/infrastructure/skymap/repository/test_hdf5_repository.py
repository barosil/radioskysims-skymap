from __future__ import annotations

from pathlib import Path

import healpy as hp
import numpy as np
import pytest

from radioskysims_skymap.domain.skymap import (
    NSide,
    SkyMap,
)
from radioskysims_skymap.infrastructure.skymap.repository.hdf5 import (
    HDF5SkyMapRepository,
)


def _skymap(
    *,
    value: float = 42.0,
    freq_ghz: float | None = 1.0,
) -> SkyMap:
    nside = 8

    return SkyMap(
        dataset=np.full(
            hp.nside2npix(nside),
            value,
            dtype=np.float64,
        ),
        nside=NSide(nside),
        unit="uK_CMB",
        frame="galactic",
        freq_ghz=freq_ghz,
        field="I",
    )


def test_save_creates_hdf5_file(
    tmp_path: Path,
):
    repository = HDF5SkyMapRepository(tmp_path)

    filepath = repository.save(
        "galaxy_1000mhz",
        _skymap(),
    )

    path = Path(filepath)

    assert path.exists()

    assert path.suffix == ".h5"


def test_roundtrip_preserves_skymap(
    tmp_path: Path,
):
    repository = HDF5SkyMapRepository(tmp_path)

    original = _skymap(
        value=123.0,
        freq_ghz=1.1,
    )

    repository.save(
        "galaxy_1100mhz",
        original,
    )

    loaded = repository.load("galaxy_1100mhz")

    assert loaded is not None

    assert loaded.nside_value == original.nside_value

    assert loaded.npix == original.npix

    assert loaded.unit == original.unit

    assert loaded.frame == original.frame

    assert loaded.field == original.field

    assert loaded.freq_ghz == original.freq_ghz

    np.testing.assert_array_equal(
        loaded.data,
        original.data,
    )


def test_roundtrip_preserves_none_frequency(
    tmp_path: Path,
):
    repository = HDF5SkyMapRepository(tmp_path)

    original = _skymap(freq_ghz=None)

    repository.save(
        "frequency_unknown",
        original,
    )

    loaded = repository.load("frequency_unknown")

    assert loaded is not None

    assert loaded.freq_ghz is None


def test_load_missing_returns_none(
    tmp_path: Path,
):
    repository = HDF5SkyMapRepository(tmp_path)

    assert repository.load("does_not_exist") is None


def test_exists_and_delete(
    tmp_path: Path,
):
    repository = HDF5SkyMapRepository(tmp_path)

    repository.save(
        "galaxy",
        _skymap(),
    )

    assert repository.exists("galaxy")

    assert repository.delete("galaxy")

    assert not repository.exists("galaxy")

    assert not repository.delete("galaxy")


def test_save_rejects_existing_key_without_overwrite(
    tmp_path: Path,
):
    repository = HDF5SkyMapRepository(tmp_path)

    repository.save(
        "galaxy",
        _skymap(value=1.0),
    )

    with pytest.raises(FileExistsError):
        repository.save(
            "galaxy",
            _skymap(value=2.0),
        )


def test_save_can_overwrite(
    tmp_path: Path,
):
    repository = HDF5SkyMapRepository(tmp_path)

    repository.save(
        "galaxy",
        _skymap(value=1.0),
    )

    repository.save(
        "galaxy",
        _skymap(value=2.0),
        overwrite=True,
    )

    loaded = repository.load("galaxy")

    assert loaded is not None

    np.testing.assert_allclose(
        loaded.data,
        2.0,
    )


def test_repository_rejects_unsafe_key(
    tmp_path: Path,
):
    repository = HDF5SkyMapRepository(tmp_path)

    with pytest.raises(ValueError):
        repository.save(
            "../outside",
            _skymap(),
        )
