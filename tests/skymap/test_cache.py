# tests/infrastructure/skymap/io/test_cache.py

from pathlib import Path

from radioskysims_skymap.infrastructure.skymap.io.cache import CacheStore


def test_cache_save_and_load(tmp_path: Path):
    cache = CacheStore(tmp_path)

    payload = {"freq": 30.0, "nside": 1}
    data = b"hello skymap"

    path = cache.save(payload, "bin", data)
    assert path.exists()

    loaded_path = cache.load(payload, "bin")
    assert loaded_path.read_bytes() == data
