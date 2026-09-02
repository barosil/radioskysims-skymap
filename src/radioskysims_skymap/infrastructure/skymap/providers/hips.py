"""\
Módulo: src/radioskysims/infrastructure/skymap/providers/hips.py

Adapter de infraestrutura para mapas contínuos via HiPS/hips2fits.

Este módulo implementa a porta ``ContinuumSurveyMapPort`` falando com
datasets HiPS expostos via um serviço ``hips2fits`` (que pode rodar
localmente no observatório ou em um endpoint remoto) **ou** usando
arquivos FITS HEALPix já gerados e armazenados em disco.

O fluxo canônico pensado é:

1. Um repositório (banco) mantém metadados dos surveys contínuos:
    ``survey_id`` (HiPS ID), URL do serviço ``hips2fits``, unidades, etc.
2. A infraestrutura deste módulo consulta esse repositório (ou recebe
    essas informações via configuração) e faz o *fetch* do cutout usando
    o serviço ``hips2fits``.
3. O resultado FITS (tipicamente uma imagem projetada) é reprojetado
    para um mapa HEALPix 1D com ``NSide`` alvo, retornando um ``SkyMap``
    pronto para uso em TOD/Simulation.

Também suportamos, como fallback, o caso em que o cutout HEALPix já foi
gerado offline (por exemplo, via hips2fits + reproject) e salvo em
disco; nesse caso usamos diretamente o FITS local.

Copyright BINGO Collaboration
Last modified: 2026-02-26
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import healpy as hp
import numpy as np
from astropy.io import fits
from reproject import reproject_to_healpix

from radioskysims_skymap.application.skymap.ports import (
    ContinuumSurveyMapPort,
)
from radioskysims_skymap.domain.skymap import NSide, SkyMap


def _is_numeric_field(dtype) -> bool:
    """True se o dtype (ou a base de um subarray) é numérico.

    FITS de tabelas HEALPix escritos pelo healpy usam colunas com dtype de
    subarray (ex.: ``('>f8', (npix,))``) — ``np.issubdtype`` direto retorna
    False; usa a base escalar.
    """
    try:
        base = np.dtype(dtype).base
    except Exception:  # noqa: BLE001 - dtype mal formado
        base = np.dtype(dtype)
    return bool(np.issubdtype(base, np.number))


class HiPSSkyMapProvider(ContinuumSurveyMapPort):
    """Adapter de HiPS/hips2fits para ``SkyMap``.

    O provider opera inicialmente com arquivos FITS locais.

    São suportados dois formatos:

    1. FITS contendo mapa HEALPix, identificado pela presença da
       palavra-chave ``NSIDE`` no cabeçalho;
    2. FITS contendo imagem bidimensional com WCS, como os recortes
       produzidos pelo serviço HiPS2FITS.

    No primeiro caso, a resolução é ajustada com ``healpy.ud_grade``.
    No segundo caso, a imagem é reprojetada para HEALPix usando
    ``reproject_to_healpix``.
    """

    def __init__(
        self,
        *,
        base_path: str | Path | None = None,
    ) -> None:
        """Inicializa o provider.

        Parameters
        ----------
        base_path:
            Diretório que contém os arquivos FITS locais. Cada survey
            é resolvido inicialmente como ``<base_path>/<survey_id>.fits``.
        """

        self._base_path = (
            Path(base_path)
            if base_path is not None
            else None
        )

    def _resolve_local_fits(
        self,
        survey_id: str,
    ) -> Path:
        """Resolve um identificador de survey para um FITS local."""

        if self._base_path is None:
            msg = (
                "HiPSSkyMapProvider requer base_path para "
                "resolver arquivos locais."
            )
            raise ValueError(
                msg
            )

        path = self._base_path / f"{survey_id}.fits"

        if not path.is_file():
            msg = (
                "Arquivo FITS não encontrado para "
                f"survey_id={survey_id!r}: {path}"
            )
            raise FileNotFoundError(
                msg
            )

        return path

    @staticmethod
    def _select_data_hdu(
        hdul: fits.HDUList,
    ) -> Any:
        """Seleciona a HDU científica apropriada.

        Prioriza uma extensão que contenha ``NSIDE``. Na ausência
        dela, retorna a primeira HDU que possua dados.
        """

        for hdu in hdul:
            if (
                hdu.data is not None
                and "NSIDE" in hdu.header
            ):
                return hdu

        for hdu in hdul:
            if hdu.data is not None:
                return hdu

        msg = "O arquivo FITS não contém nenhuma HDU com dados."
        raise ValueError(
            msg
        )

    @staticmethod
    def _extract_healpix_vector(
        data: Any,
        *,
        field: str,
    ) -> np.ndarray:
        """Extrai um vetor numérico de um FITS HEALPix.

        FITS HEALPix podem armazenar os dados diretamente como array
        ou dentro de uma tabela binária com uma ou mais colunas.
        """

        raw_data = np.asarray(data)

        if raw_data.dtype.names:
            names = raw_data.dtype.names

            names_by_upper = {
                name.upper(): name
                for name in names
            }

            candidates = (
                field,
                f"{field}_STOKES",
                # Nomes comuns de colunas HEALPix (healpy grava "T" por padrão)
                "T",
                "I",
                "Q",
                "U",
                "TEMPERATURE",
                "SIGNAL",
                "SKYMAP",
                "VALUE",
                "MAP",
            )

            column_name = next(
                (
                    names_by_upper[candidate.upper()]
                    for candidate in candidates
                    if candidate.upper() in names_by_upper
                ),
                None,
            )

            if column_name is None:
                numeric_columns = [
                    name
                    for name in names
                    if _is_numeric_field(
                        raw_data.dtype.fields[name][0]
                    )
                ]

                if not numeric_columns:
                    msg = (
                        "A tabela FITS HEALPix não possui "
                        "colunas numéricas."
                    )
                    raise ValueError(
                        msg
                    )

                column_name = numeric_columns[0]

            raw_data = raw_data[column_name]

        vector = np.asarray(
            raw_data,
            dtype=np.float64,
        ).reshape(-1)

        if vector.size == 0:
            msg = "O FITS HEALPix contém um dataset vazio."
            raise ValueError(
                msg
            )

        return vector

    @staticmethod
    def _frame_from_header(
        header: fits.Header,
    ) -> str:
        """Infere o referencial de um mapa HEALPix pelo cabeçalho."""

        coord_system = str(
            header.get(
                "COORDSYS",
                header.get("COORDTYPE", "G"),
            )
        ).strip().upper()

        if coord_system.startswith("C"):
            return "equatorial"

        if coord_system.startswith("E"):
            return "ecliptic"

        return "galactic"

    @staticmethod
    def _convert_healpix_map(
        *,
        data: Any,
        header: fits.Header,
        target_nside: NSide,
        field: str,
    ) -> np.ndarray:
        """Normaliza e altera a resolução de um mapa HEALPix."""

        source_nside = int(header["NSIDE"])

        healpix_data = (
            HiPSSkyMapProvider._extract_healpix_vector(
                data,
                field=field,
            )
        )

        expected_source_npix = hp.nside2npix(
            source_nside
        )

        if healpix_data.size != expected_source_npix:
            msg = (
                "Quantidade de pixels incompatível com o NSIDE "
                f"do FITS: esperado={expected_source_npix}, "
                f"recebido={healpix_data.size}."
            )
            raise ValueError(
                msg
            )

        ordering = str(
            header.get("ORDERING", "RING")
        ).strip().upper()

        order_in = (
            "NESTED"
            if ordering.startswith("NEST")
            else "RING"
        )

        if source_nside == target_nside.value:
            if order_in == "NESTED":
                result = hp.reorder(
                    healpix_data,
                    n2r=True,
                )
            else:
                result = healpix_data.copy()
        else:
            result = hp.ud_grade(
                healpix_data,
                nside_out=target_nside.value,
                order_in=order_in,
                order_out="RING",
            )

        return np.asarray(
            result,
            dtype=np.float64,
        ).reshape(-1)

    @staticmethod
    def _convert_wcs_image(
        *,
        data: Any,
        header: fits.Header,
        target_nside: NSide,
    ) -> np.ndarray:
        """Reprojeta uma imagem WCS bidimensional para HEALPix."""

        image_data = np.squeeze(
            np.asarray(
                data,
                dtype=np.float64,
            )
        )

        if image_data.ndim != 2:
            msg = (
                "O FITS projetado deve conter uma imagem WCS "
                "bidimensional; "
                f"shape recebido: {image_data.shape}."
            )
            raise ValueError(
                msg
            )

        if not np.any(np.isfinite(image_data)):
            msg = "A imagem WCS não contém pixels finitos."
            raise ValueError(
                msg
            )

        skymap_data, footprint = reproject_to_healpix(
            (image_data, header),
            coord_system_out="galactic",
            nested=False,
            nside=target_nside.value,
        )

        skymap_data = np.asarray(
            skymap_data,
            dtype=np.float64,
        ).reshape(-1)

        footprint = np.asarray(
            footprint,
            dtype=np.float64,
        ).reshape(-1)

        expected_npix = target_nside.npix

        if skymap_data.size != expected_npix:
            msg = (
                "A reprojeção produziu quantidade inesperada "
                f"de pixels: esperado={expected_npix}, "
                f"recebido={skymap_data.size}."
            )
            raise ValueError(
                msg
            )

        valid_pixels = (
            np.isfinite(skymap_data)
            & np.isfinite(footprint)
            & (footprint > 0.0)
        )

        if not np.any(valid_pixels):
            resolution_deg = float(
                np.degrees(
                    hp.nside2resol(
                        target_nside.value
                    )
                )
            )

            msg = (
                "A reprojeção WCS para HEALPix não produziu "
                "pixels válidos. Isso pode ocorrer quando o "
                "recorte WCS é menor que a resolução dos pixels "
                "HEALPix ou quando o cabeçalho WCS é inválido. "
                f"NSide={target_nside.value}, "
                f"resolução aproximada={resolution_deg:.3f} graus."
            )
            raise ValueError(
                msg
            )
        # O SkyMap atual não carrega uma máscara ou footprint junto
        # ao dataset. Durante esta implementação inicial, regiões
        # externas ao recorte são preenchidas com zero.
        return np.where(
            valid_pixels,
            skymap_data,
            0.0,
        ).astype(
            np.float64,
            copy=False,
        )

    @staticmethod
    def _extract_unit(
        header: fits.Header,
    ) -> str:
        """Extrai a unidade física do cabeçalho FITS."""

        unit = str(
            header.get(
                "BUNIT",
                header.get("TUNIT1", "unknown"),
            )
        ).strip()

        return unit or "unknown"

    def fetch_cutout(
        self,
        *,
        survey_id: str,
        ra_deg: float,
        dec_deg: float,
        fov_deg: float,
        nside: NSide,
        freq_ghz: float | None = None,
        field: str = "I",
    ) -> SkyMap:
        """Obtém um mapa contínuo a partir de um FITS local.

        Nesta implementação inicial, ``ra_deg``, ``dec_deg`` e
        ``fov_deg`` descrevem o recorte que originou o arquivo FITS,
        mas o provider ainda não realiza a consulta remota diretamente.

        O arquivo local é resolvido como:

        ``<base_path>/<survey_id>.fits``
        """

        # Mantidos no contrato para futura consulta direta ao HiPS.
        _ = (
            ra_deg,
            dec_deg,
            fov_deg,
        )

        fits_path = self._resolve_local_fits(
            survey_id
        )

        with fits.open(
            fits_path,
            memmap=False,
        ) as hdul:
            hdu = self._select_data_hdu(hdul)

            data = np.array(
                hdu.data,
                copy=True,
            )

            header = hdu.header.copy()

        source_nside = header.get("NSIDE")

        if source_nside is not None:
            skymap_data = self._convert_healpix_map(
                data=data,
                header=header,
                target_nside=nside,
                field=field,
            )

            frame = self._frame_from_header(
                header
            )
        else:
            skymap_data = self._convert_wcs_image(
                data=data,
                header=header,
                target_nside=nside,
            )

            frame = "galactic"

        return SkyMap(
            dataset=np.asarray(
                skymap_data,
                dtype=np.float64,
            ).reshape(-1),
            nside=nside,
            unit=self._extract_unit(header),
            frame=frame,
            freq_ghz=freq_ghz,
            field=field,
        )
