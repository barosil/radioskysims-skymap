"""\
Módulo: tests/application/orchestrators/test_skymap_orchestrator.py

Descrição:
    Testes da fachada ``SkyMapOrchestrator`` (UX Onda 3/R6) — fluxo offline
    (gaussiano) sem PySM3.

Copyright BINGO Collaboration
"""

from __future__ import annotations

import numpy as np
import pytest
from radioskysims_skymap import Mask, SkyMapOrchestrator, SkyMask


def test_from_noise_gaussian_offline():
    orch = SkyMapOrchestrator().from_noise(nside=4, method="gaussian", seed=1)
    assert orch.nside == 4
    assert orch.data.shape == (12 * 4**2,)
    assert orch.handle is not None


def test_apply_mask_e_smooth_encadeiam():
    orch = SkyMapOrchestrator().from_noise(nside=4, method="gaussian", seed=2)
    ones = np.ones(orch.data.shape)
    orch.apply(ones)
    assert orch.data.shape == ones.shape
    orch.smooth(fwhm_deg=5.0)
    assert orch.nside == 4


def test_statistics():
    orch = SkyMapOrchestrator().from_noise(nside=4, method="gaussian", seed=3)
    stats = orch.statistics()
    assert stats.mean is not None
    assert stats.rms is not None


def test_operar_sem_mapa_levanta_erro_claro():
    orch = SkyMapOrchestrator()
    with pytest.raises(ValueError):
        orch.smooth(fwhm_deg=1.0)


def test_mask_exposto_como_alias():
    assert Mask is SkyMask


def test_integracoes_pendentes():
    orch = SkyMapOrchestrator().from_noise(nside=4, method="gaussian", seed=1)
    with pytest.raises(NotImplementedError):
        orch.from_sources()
    with pytest.raises(NotImplementedError):
        orch.to_tod()
