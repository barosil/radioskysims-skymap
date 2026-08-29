from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final

import healpy as hp
import numpy as np

from radioskysims_skymap.domain.skymap.models import SkyMap
from radioskysims_skymap.domain.skymap.value_objects import NSide
from radioskysims_skymap.infrastructure.ports.skymap import (
    SkyMapIOPort,
    SkyMapTransformPort,
)
from radioskysims_skymap.infrastructure.skymap.telemetry import Telemetry

_telemetry = Telemetry(__name__)

_FIELD_INDEX: Final[Mapping[str, int]] = {
    "I": 0,
    "Q": 1,
    "U": 2,
}


class HealpySkyMapProvider(SkyMapIOPort, SkyMapTransformPort):
    """
    Provider de infraestrutura baseado em HEALPy.

    Responsável por operações comuns de IO e transformações físicas
    em mapas HEALPix.
    """

    def read_map(
        self,
        path: str | Path,
        *,
        nside: NSide | None = None,
        unit: str = "unknown",
        frame: str = "galactic",
        field: str = "I",
    ) -> SkyMap:
        with _telemetry.measure("read_map"):
            field_index = _FIELD_INDEX.get(field)
            if field_index is None:
                msg = "Field must be I, Q or U"
                raise ValueError(msg)

            map_data = hp.read_map(Path(path), field=field_index)
            data = np.asarray(map_data)

            # Garantir que resolved_nside é sempre um NSide object
            if nside is not None:
                resolved_nside = NSide(nside) if not isinstance(nside, NSide) else nside
            else:
                resolved_nside = NSide(hp.get_nside(data))

            result = SkyMap(
                dataset=data,
                nside=resolved_nside,
                unit=unit,
                frame=frame,
                field=field,
            )

            _telemetry.event(
                "map_read",
                npix=len(data),
                nside=resolved_nside.value,
                unit=unit,
                field=field,
            )
            return result

    def read_header(self, path: str | Path) -> dict[str, object]:
        _, header = hp.read_map(Path(path), h=True)
        return dict(header)

    def write_map(
        self,
        skymap: SkyMap,
        path: str | Path,
        *,
        overwrite: bool = False,
        dtype: type | None = float,
    ) -> Path:
        with _telemetry.measure("write_map"):
            output_path = Path(path)
            data = skymap.data if dtype is None else skymap.data.astype(dtype)
            hp.write_map(output_path, data, overwrite=overwrite)

            _telemetry.event(
                "map_written",
                filepath=str(output_path),
                npix=len(data),
                nside=skymap.nside.value,
            )
            return output_path

    def smooth(self, skymap: SkyMap, fwhm_rad: float) -> SkyMap:
        with _telemetry.measure("smooth"):
            if fwhm_rad <= 0:
                msg = "fwhm_rad must be positive"
                raise ValueError(msg)

            smoothed = hp.smoothing(skymap.data, fwhm=fwhm_rad)
            result = SkyMap(
                dataset=smoothed,
                nside=skymap.nside,
                unit=skymap.unit,
                frame=skymap.frame,
                freq_ghz=skymap.freq_ghz,
                field=skymap.field,
            )

            _telemetry.event(
                "map_smoothed",
                npix=len(smoothed),
                fwhm_rad=fwhm_rad,
            )
            return result

    def rotate(
        self,
        skymap: SkyMap,
        rot: tuple[float, float, float],
    ) -> SkyMap:
        with _telemetry.measure("rotate"):
            alm = hp.map2alm(skymap.data)
            psi, theta, phi = rot
            hp.rotate_alm(alm, psi=psi, theta=theta, phi=phi)  # modifica alm in-place
            rotated = hp.alm2map(alm, hp.get_nside(skymap.data))

            result = SkyMap(
                dataset=rotated,
                nside=skymap.nside,
                unit=skymap.unit,
                frame=skymap.frame,
                freq_ghz=skymap.freq_ghz,
                field=skymap.field,
            )

            _telemetry.event(
                "map_rotated",
                npix=len(rotated),
                rot_angles=rot,
            )
            return result

    def downgrade(self, skymap: SkyMap, target_nside: NSide) -> SkyMap:
        with _telemetry.measure("downgrade"):
            downgraded = hp.ud_grade(
                skymap.data,
                nside_out=target_nside.value,
                order_in="RING",
                order_out="RING",
                power=-2,
            )

            result = SkyMap(
                dataset=downgraded,
                nside=target_nside,
                unit=skymap.unit,
                frame=skymap.frame,
                freq_ghz=skymap.freq_ghz,
                field=skymap.field,
            )

            _telemetry.event(
                "map_downgraded",
                npix=len(downgraded),
                target_nside=target_nside.value,
            )
            return result
