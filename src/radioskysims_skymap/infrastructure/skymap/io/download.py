# src/radioskysims/infrastructure/skymap/io/download.py

from __future__ import annotations

from pathlib import Path
import requests


class DownloadError(RuntimeError):
    pass


class FileDownloader:
    """
    Responsável apenas por baixar arquivos externos.
    """

    def __init__(self, timeout: int = 30):
        self._timeout = timeout

    def download(
        self,
        url: str,
        destination: Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        """
        Baixa um arquivo para o caminho indicado.
        """
        destination = Path(destination)

        if destination.exists() and not overwrite:
            return destination

        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            response = requests.get(url, timeout=self._timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise DownloadError(f"Failed to download {url}") from exc

        destination.write_bytes(response.content)
        return destination
