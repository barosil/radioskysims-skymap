"""\
Módulo: infrastructure/skymap/providers/pysm_templates.py

Gerenciador de download/cache dos templates PySM3.

PySM3 baixa os templates de componentes (synchrotron, dust, free-free, AME,
CMB) sob demanda via ``pysm3.utils.data.RemoteData`` (cache do astropy). Este
adapter expõe esse download como operação explícita: resolve os arquivos do
``presets.cfg`` do PySM3, baixa cada um (com progresso) e os copia para um
diretório local, definindo ``PYSM_LOCAL_DATA`` para que as gerações seguintes
usem os arquivos locais (sem rede).

Copyright BINGO Collaboration
Last modified: 2026-09-01
"""

from __future__ import annotations

import configparser
import os
import shutil
from pathlib import Path

from radioskysims_skymap.domain.skymap.exceptions import SkyMapError

# Mapeamento componente → preset PySM3 (presets.cfg).
_COMPONENT_PRESET: dict[str, str] = {
    "synchrotron": "s1",
    "dust": "d1",
    "freefree": "f1",
    "ame": "a1",
    "cmb": "c1",
}

# Extensões reconhecidas como arquivos de template (vs. escalares).
_TEMPLATE_EXTENSIONS: tuple[str, ...] = (
    ".fits",
    ".dat",
    ".txt",
    ".npy",
    ".h5",
)


class PySMTemplateError(SkyMapError):
    """Falha ao baixar/gerenciar templates PySM3."""


class PySMTemplateManager:
    """Baixa/cacheia templates PySM3 de componentes num diretório local."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        """Inicializa o gerenciador.

        Args:
            data_dir: Diretório local para os templates. Padrão ``data/pysm``.
        """
        self.data_dir = Path(data_dir or "data/pysm")
        self._presets = self._load_presets()

    @staticmethod
    def _load_presets() -> configparser.ConfigParser:
        """Carrega o ``presets.cfg`` do pacote PySM3 instalado."""
        import pysm3

        cfg = configparser.ConfigParser()
        path = Path(pysm3.__file__).parent / "data" / "presets.cfg"
        if not path.exists():
            msg = f"presets.cfg do PySM3 não encontrado: {path}"
            raise PySMTemplateError(msg)
        cfg.read(path)
        return cfg

    @classmethod
    def preset_for(cls, component: str) -> str:
        """Preset PySM3 de um componente físico."""
        try:
            return _COMPONENT_PRESET[component]
        except KeyError:
            msg = f"componente desconhecido: {component!r} (use {list(_COMPONENT_PRESET)})"
            raise ValueError(msg) from None

    def template_filenames(self, components: list[str]) -> list[str]:
        """Arquivos de template necessários para os componentes.

        Resolve do ``presets.cfg`` os valores que são caminhos de arquivo
        (extensões conhecidas), ignorando escalares (ex.: ``-2.14``) e
        unidades (ex.: ``"408 MHz"``).
        """
        files: list[str] = []
        for component in components:
            preset = self.preset_for(component)
            if not self._presets.has_section(preset):
                # ex.: a1 (ame) pode não ter seção no presets.cfg instalado
                continue
            for _, value in self._presets.items(preset):
                candidate = value.strip().strip('"')
                if candidate.lower().endswith(_TEMPLATE_EXTENSIONS):
                    files.append(candidate)
        # dedup preservando ordem
        return list(dict.fromkeys(files))

    def is_available(self, components: list[str]) -> bool:
        """True se todos os templates dos componentes já estão locais."""
        return all(
            (self.data_dir / filename).exists() for filename in self.template_filenames(components)
        )

    def download(
        self,
        components: list[str],
        *,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> list[Path]:
        """Baixa (ou usa cache) os templates dos componentes para ``data_dir``.

        Usa o próprio mecanismo do PySM3 (``RemoteData`` → cache astropy) e
        copia os arquivos para ``data_dir``, definindo ``PYSM_LOCAL_DATA``
        para as gerações seguintes.

        Returns
        -------
        list[Path]
            Caminhos locais dos templates.
        """
        from pysm3.utils.data import RemoteData

        self.data_dir.mkdir(parents=True, exist_ok=True)
        remote = RemoteData()
        downloaded: list[Path] = []
        for filename in self.template_filenames(components):
            target = self.data_dir / filename
            if target.exists() and not overwrite:
                downloaded.append(target)
                continue
            source = remote.get(filename)  # baixa via astropy cache (progresso)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            downloaded.append(target)
            if verbose:
                size_mb = target.stat().st_size / 1e6
                print(f"[pysm] template {filename} ({size_mb:.1f} MB)")
        # gerações futuras usam os arquivos locais
        os.environ["PYSM_LOCAL_DATA"] = str(self.data_dir)
        return downloaded


def download_pysm_templates(
    components: list[str],
    data_dir: str | Path | None = None,
    *,
    verbose: bool = True,
) -> list[Path]:
    """Conveniência: baixa templates PySM3 de componentes (cache local)."""
    return PySMTemplateManager(data_dir=data_dir).download(components, verbose=verbose)
