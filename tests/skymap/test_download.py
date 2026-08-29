# tests/infrastructure/skymap/io/test_download.py

from pathlib import Path
from unittest.mock import patch, Mock

from radioskysims_skymap.infrastructure.skymap.io.download import FileDownloader


def test_download_with_mock(tmp_path: Path):
    downloader = FileDownloader()

    fake_response = Mock()
    fake_response.content = b"data"
    fake_response.raise_for_status = Mock()

    with patch("requests.get", return_value=fake_response):
        path = downloader.download(
            "http://example.com/file.bin",
            tmp_path / "file.bin",
        )

    assert path.exists()
    assert path.read_bytes() == b"data"
