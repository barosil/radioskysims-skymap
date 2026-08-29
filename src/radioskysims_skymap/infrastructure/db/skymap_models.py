from __future__ import annotations

from datetime import datetime
from io import BytesIO

import numpy as np
from sqlalchemy import Column, LargeBinary
from sqlmodel import Field, SQLModel

from radioskysims_skymap.domain.skymap import NSide, SkyMap


class SkyMapRecord(SQLModel, table=True):  # type: ignore[call-arg, misc]
    """
    Persistência de SkyMap em banco (SQLModel).

    Armazena o array via numpy.save em blob binário.
    """

    id: int | None = Field(default=None, primary_key=True)
    nside: int = Field(index=True)
    npix: int
    unit: str
    frame: str
    field: str
    freq_ghz: float | None = None
    data_blob: bytes = Field(sa_column=Column(LargeBinary, nullable=False))
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @staticmethod
    def _encode_array(data: np.ndarray) -> bytes:
        buffer = BytesIO()
        np.save(buffer, data, allow_pickle=False)
        return buffer.getvalue()

    @staticmethod
    def _decode_array(data_blob: bytes) -> np.ndarray:
        buffer = BytesIO(data_blob)
        return np.asarray(np.load(buffer, allow_pickle=False))

    @classmethod
    def from_skymap(cls, skymap: SkyMap) -> SkyMapRecord:
        data = skymap.data
        return cls(
            nside=skymap.nside_value,
            npix=skymap.npix,
            unit=skymap.unit,
            frame=skymap.frame,
            field=skymap.field,
            freq_ghz=skymap.freq_ghz,
            data_blob=cls._encode_array(data),
        )

    def to_skymap(self) -> SkyMap:
        data = self._decode_array(self.data_blob)
        return SkyMap(
            dataset=data,
            nside=NSide(self.nside),
            unit=self.unit,
            frame=self.frame,
            freq_ghz=self.freq_ghz,
            field=self.field,
        )
