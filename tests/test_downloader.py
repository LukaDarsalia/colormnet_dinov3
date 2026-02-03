from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

from src.downloader.utils import extract_archive, stage_download


def _create_zip(path: Path, base: str, filename: str, content: bytes) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{base}/{filename}", content)


def _create_tar(path: Path, base: str, filename: str, content: bytes) -> None:
    data = io.BytesIO(content)
    info = tarfile.TarInfo(name=f"{base}/{filename}")
    info.size = len(content)
    with tarfile.open(path, "w:gz") as tf:
        tf.addfile(info, data)


def test_extract_zip_with_strip_components(tmp_path: Path) -> None:
    archive = tmp_path / "sample.zip"
    _create_zip(archive, "top", "file.txt", b"hello")

    out_dir = tmp_path / "out"
    extract_archive(archive, out_dir, strip_components=1)
    assert (out_dir / "file.txt").read_text() == "hello"


def test_extract_tar_with_strip_components(tmp_path: Path) -> None:
    archive = tmp_path / "sample.tar.gz"
    _create_tar(archive, "top", "file.txt", b"hello")

    out_dir = tmp_path / "out"
    extract_archive(archive, out_dir, strip_components=1)
    assert (out_dir / "file.txt").read_text() == "hello"


def test_stage_download_local_file(tmp_path: Path) -> None:
    src = tmp_path / "src.txt"
    src.write_text("data")
    cache = tmp_path / "cache"
    result = stage_download(str(src), cache)
    assert result.resolve() == src.resolve()
