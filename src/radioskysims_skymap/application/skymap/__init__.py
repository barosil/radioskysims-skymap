# application/skymap/dto.py
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class GenerateSkyMapRequest:
    nside: int
    components: Sequence[str]


@dataclass(frozen=True)
class GenerateSkyMapResponse:
    nside: int
    components: Sequence[str]
    npix: int
