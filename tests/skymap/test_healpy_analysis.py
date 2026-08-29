# tests/skymap/test_healpy_analysis.py

import numpy as np

from radioskysims_skymap.infrastructure.skymap.adapters.analysis import (
    HealpySkyMapAnalyzerAdapter,
)
from radioskysims_skymap.domain.skymap.models import SkyMap
from radioskysims_skymap.domain.skymap.value_objects import NSide


def test_compute_cls():
    data = np.ones(12)

    skymap = SkyMap(
        dataset=data,
        nside=NSide(1),
        unit="uK_CMB",
    )

    analyzer = HealpySkyMapAnalyzerAdapter()

    cls = analyzer.compute_cls(
        map_data=skymap.data,
        lmax=3,
    )

    assert isinstance(cls, np.ndarray)
    assert cls.ndim == 1
