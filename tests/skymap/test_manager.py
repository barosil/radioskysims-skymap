from __future__ import annotations

from unittest.mock import MagicMock, patch

import healpy as hp
import numpy as np
import pytest
from radioskysims_skymap.domain.skymap.models import SkyMap
from radioskysims_skymap.domain.skymap.value_objects import NSide
from radioskysims_skymap.manager import (
    SkyMapHandle,
    SkyMapManager,
    SkyMapStatistics,
)

NSIDE = 8


def make_handle(*, frame: str = "galactic", seed: int = 1) -> SkyMapHandle:
    rng = np.random.default_rng(seed)
    data = rng.normal(size=hp.nside2npix(NSIDE))
    skymap = SkyMap(
        dataset=data,
        nside=NSide(NSIDE),
        unit="uK_CMB",
        frame=frame,
        freq_ghz=1.0,
        field="I",
    )
    return SkyMapHandle(skymap)


# ---------------------------------------------------------------------
# SkyMapManager.create_map_from_coefficients
# ---------------------------------------------------------------------


def test_create_map_from_coefficients_returns_handle() -> None:
    ell = np.arange(3, 20)
    cls_input = 1.0 * ell**-1.5

    handle = SkyMapManager.create_map_from_coefficients(
        cls_input,
        nside=NSIDE,
        lmax=19,
        seed=42,
        unit="uK_CMB",
        freq_ghz=1.0,
    )

    assert isinstance(handle, SkyMapHandle)
    assert handle.nside_value == NSIDE
    assert handle.unit == "uK_CMB"
    assert handle.data.size == hp.nside2npix(NSIDE)


def test_create_map_from_coefficients_is_deterministic_with_seed() -> None:
    ell = np.arange(3, 20)
    cls_input = 1.0 * ell**-1.5

    first = SkyMapManager.create_map_from_coefficients(
        cls_input,
        nside=NSIDE,
        lmax=19,
        seed=7,
    )
    second = SkyMapManager.create_map_from_coefficients(
        cls_input,
        nside=NSIDE,
        lmax=19,
        seed=7,
    )

    np.testing.assert_array_equal(first.data, second.data)


# ---------------------------------------------------------------------
# SkyMapManager.create_noise_map
# ---------------------------------------------------------------------


def test_create_noise_map_gaussian() -> None:
    handle = SkyMapManager.create_noise_map(
        nside=NSIDE,
        method="gaussian",
        seed=1,
    )

    assert isinstance(handle, SkyMapHandle)
    assert handle.data.size == hp.nside2npix(NSIDE)


def test_create_noise_map_pysm_delegates_to_adapter() -> None:
    fake_skymap = make_handle().skymap

    with patch("radioskysims_skymap.manager.PySMSkyMapGeneratorAdapter") as mock_cls:
        mock_cls.return_value.generate.return_value = fake_skymap

        handle = SkyMapManager.create_noise_map(
            nside=NSIDE,
            freq_ghz=30.0,
            method="pysm",
            components=["synchrotron"],
        )

        mock_cls.return_value.generate.assert_called_once()
        assert handle.skymap is fake_skymap


def test_create_noise_map_pysm_requires_freq_ghz() -> None:
    with pytest.raises(ValueError):
        SkyMapManager.create_noise_map(nside=NSIDE, method="pysm")


def test_create_noise_map_invalid_method_raises() -> None:
    with pytest.raises(ValueError):
        SkyMapManager.create_noise_map(nside=NSIDE, method="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# I/O: FITS e HDF5
# ---------------------------------------------------------------------


def test_save_and_load_fits_roundtrip(tmp_path) -> None:
    handle = make_handle()
    path = tmp_path / "map.fits"

    saved_path = handle.save_to_file(path, format="fits", overwrite=True)
    assert saved_path.exists()

    reloaded = SkyMapManager.load_map_from_file(
        saved_path,
        unit=handle.unit,
        frame=handle.frame,
    )

    np.testing.assert_allclose(reloaded.data, handle.data)


def test_save_to_file_hdf5_and_register_roundtrip(tmp_path) -> None:
    handle = make_handle()

    saved_path = handle.save_to_file(
        tmp_path / "map.h5",
        format="hdf5",
        overwrite=True,
    )
    assert saved_path == tmp_path / "map.h5"

    handle.register("my-map", root=tmp_path / "registry")
    reloaded = SkyMapManager.load_registered("my-map", root=tmp_path / "registry")

    assert reloaded is not None
    np.testing.assert_allclose(reloaded.data, handle.data)


def test_load_registered_returns_none_when_missing(tmp_path) -> None:
    result = SkyMapManager.load_registered("missing", root=tmp_path / "registry")
    assert result is None


def test_save_to_file_invalid_format_raises(tmp_path) -> None:
    handle = make_handle()

    with pytest.raises(ValueError):
        handle.save_to_file(tmp_path / "map.xyz", format="xyz")  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# Operações encadeáveis
# ---------------------------------------------------------------------


def test_smooth_returns_new_handle_and_preserves_original() -> None:
    handle = make_handle()
    smoothed = handle.smooth(fwhm_deg=5.0)

    assert isinstance(smoothed, SkyMapHandle)
    assert smoothed is not handle
    assert not np.array_equal(smoothed.data, handle.data)


def test_mask_applies_binary_mask() -> None:
    handle = make_handle()
    mask = np.ones_like(handle.data)
    mask[len(mask) // 2 :] = 0.0

    masked = handle.mask(mask)

    np.testing.assert_allclose(masked.data, handle.data * mask)


def test_downgrade_reduces_nside() -> None:
    handle = make_handle()
    downgraded = handle.downgrade(4)

    assert downgraded.nside_value == 4
    assert downgraded.data.size == hp.nside2npix(4)


def test_rotate_with_rot_deg() -> None:
    handle = make_handle()
    rotated = handle.rotate(rot_deg=(10.0, 0.0, 0.0))

    assert isinstance(rotated, SkyMapHandle)
    assert rotated.frame == handle.frame


def test_rotate_requires_rot_deg_or_target_frame() -> None:
    handle = make_handle()

    with pytest.raises(ValueError):
        handle.rotate()


def test_change_coordinates_updates_frame_and_data() -> None:
    handle = make_handle(frame="galactic")
    converted = handle.change_coordinates("equatorial")

    assert converted.frame == "equatorial"
    assert not np.array_equal(converted.data, handle.data)


def test_change_coordinates_same_frame_is_noop() -> None:
    handle = make_handle(frame="galactic")
    same = handle.change_coordinates("galactic")

    assert same is handle


def test_change_coordinates_invalid_frame_raises() -> None:
    handle = make_handle()

    with pytest.raises(ValueError):
        handle.change_coordinates("bogus")


def test_chained_operations() -> None:
    handle = make_handle()
    mask = np.ones_like(handle.data)

    result = handle.smooth(fwhm_deg=2.0).mask(mask)

    assert isinstance(result, SkyMapHandle)
    assert result.data.size == handle.data.size


# ---------------------------------------------------------------------
# Análise
# ---------------------------------------------------------------------


def test_compute_statistics() -> None:
    handle = make_handle()
    stats = handle.compute_statistics()

    assert isinstance(stats, SkyMapStatistics)
    assert stats.min <= stats.mean <= stats.max


def test_compute_power_spectrum() -> None:
    handle = make_handle()
    cls = handle.compute_power_spectrum(lmax=10)

    assert isinstance(cls, np.ndarray)
    assert cls.size == 11


# ---------------------------------------------------------------------
# Visualização
# ---------------------------------------------------------------------


def test_view_delegates_to_visualizer_and_returns_self() -> None:
    handle = make_handle()
    mock_visualizer = MagicMock()
    handle_with_mock = SkyMapHandle(handle.skymap, visualizer=mock_visualizer)

    result = handle_with_mock.view(show=False)

    mock_visualizer.plot_map.assert_called_once()
    assert result is handle_with_mock
