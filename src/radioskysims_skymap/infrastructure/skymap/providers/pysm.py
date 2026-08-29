from __future__ import annotations

"""Provider para geração de SkyMap usando PySM3."""

from typing import Final

import numpy as np
import pysm3
import pysm3.units as u

from radioskysims_skymap.application.skymap.ports import SkyMapGeneratorPort
from radioskysims_skymap.domain.skymap import NSide, SkyMap

_COMPONENT_PRESET_MAP: Final[dict[str, str]] = {
    "synchrotron": "s1",
    "dust": "d1",
    "freefree": "f1",
    "ame": "a1",
    "cmb": "c1",
}


class PySMSkyMapGeneratorAdapter(SkyMapGeneratorPort):
    """
    Adapter de geração de SkyMap usando PySM3.

    Implementa a porta SkyMapGeneratorPort para gerar mapas do céu
    sintéticos usando componentes do PySM3 (Synchrotron, Dust, Free-free,
    AME, CMB).
    """

    def __init__(
        self,
        *,
        unit: str = "uK_CMB",
        field: str = "I",
    ):
        """
        Inicializa o gerador de SkyMap com PySM3.

        Args:
            unit: Unidade de saída (padrão: "uK_CMB")
            field: Campo de Stokes (I, Q, ou U; padrão: "I")
        """
        self._unit = unit
        self._field = field

    def generate(
        self,
        *,
        nside: NSide,
        freq_ghz: float,
        components: list[str],
    ) -> SkyMap:
        """
        Gera um SkyMap sintético com componentes PySM3.

        Args:
            nside: Resolução HEALPix (NSide value object)
            freq_ghz: Frequência em GHz
            components: Lista de componentes ("synchrotron", "dust", "freefree", "ame", "cmb")

        Returns:
            SkyMap com os dados gerados

        Raises:
            ValueError: Se um componente não é suportado
        """
        presets: list[str] = []

        for comp in components:
            try:
                presets.append(_COMPONENT_PRESET_MAP[comp])
            except KeyError:
                raise ValueError(f"Unsupported PySM component: {comp}") from None

        sky = pysm3.Sky(
            nside=nside.value,
            preset_strings=presets,
            output_unit=u.Unit(self._unit),
        )

        emission = sky.get_emission(freq_ghz * u.GHz).value
        # shape: (3, npix) → I, Q, U

        stokes_index = {"I": 0, "Q": 1, "U": 2}[self._field]
        map_1d = np.asarray(emission[stokes_index])

        return SkyMap(
            dataset=map_1d,
            nside=nside,
            unit=self._unit,
            freq_ghz=freq_ghz,
            field=self._field,
        )
