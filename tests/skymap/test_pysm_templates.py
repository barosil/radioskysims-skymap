"""\
Módulo: tests/skymap/test_pysm_templates.py

Descrição:
    Testes do ``PySMTemplateManager`` (download/cache de templates PySM3).

Copyright BINGO Collaboration
Last modified: 2026-09-01
"""

from __future__ import annotations

import os

import pytest

from radioskysims_skymap.infrastructure.skymap.providers.pysm_templates import (
    PySMTemplateManager,
)


@pytest.fixture
def manager(tmp_path):
    return PySMTemplateManager(data_dir=tmp_path / "pysm")


def test_preset_for(manager):
    assert manager.preset_for("freefree") == "f1"
    assert manager.preset_for("synchrotron") == "s1"
    assert manager.preset_for("dust") == "d1"
    assert manager.preset_for("cmb") == "c1"


def test_preset_unknown_raises(manager):
    with pytest.raises(ValueError, match="componente desconhecido"):
        manager.preset_for("bogus")


def test_template_filenames_freefree(manager):
    assert manager.template_filenames(["freefree"]) == ["pysm_2/ff_t_new.fits"]


def test_template_filenames_synchrotron(manager):
    files = manager.template_filenames(["synchrotron"])
    assert "pysm_2/synch_t_new.fits" in files
    assert "pysm_2/synch_beta.fits" in files


def test_template_filenames_excludes_scalars(manager):
    """Escalares (ex.: map_pl_index=-2.14) não viram arquivo."""
    files = manager.template_filenames(["freefree"])
    assert not any("2.14" in f for f in files)


def test_is_available(manager):
    assert not manager.is_available(["freefree"])
    target = manager.data_dir / "pysm_2" / "ff_t_new.fits"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"x")
    assert manager.is_available(["freefree"])


def test_download_copies_and_sets_env(tmp_path, manager, monkeypatch):
    """Download usa RemoteData (mockado) e copia p/ data_dir + PYSM_LOCAL_DATA."""
    remote_dir = tmp_path / "remote"

    class FakeRemote:
        def get(self, filename):
            src = remote_dir / filename
            src.parent.mkdir(parents=True, exist_ok=True)
            src.write_bytes(b"template")
            return src

    monkeypatch.setattr("pysm3.utils.data.RemoteData", FakeRemote)
    monkeypatch.delenv("PYSM_LOCAL_DATA", raising=False)

    paths = manager.download(["freefree"])
    assert len(paths) == 1
    assert paths[0].name == "ff_t_new.fits"
    assert paths[0].exists()
    assert os.environ["PYSM_LOCAL_DATA"] == str(manager.data_dir)
