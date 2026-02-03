from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlparse
from urllib.request import urlretrieve

import gdown


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256sum(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_gdrive_url(url: str) -> bool:
    return "drive.google.com" in url or "docs.google.com" in url


def _resolve_source_to_file(src: str, dest_dir: Path, filename: Optional[str] = None) -> Path:
    dest_dir = ensure_dir(dest_dir)
    parsed = urlparse(src)
    if parsed.scheme in ("http", "https"):
        if _is_gdrive_url(src):
            output = dest_dir / (filename or "gdrive_download")
            gdown.download(url=src, output=str(output), quiet=False, fuzzy=True)
            return output
        output = dest_dir / (filename or Path(parsed.path).name or "download")
        urlretrieve(src, output)
        return output
    if parsed.scheme == "file":
        return Path(parsed.path)
    return Path(src).expanduser().resolve()


def _strip_components(path: str, strip_components: int) -> str:
    parts = Path(path).parts
    if len(parts) <= strip_components:
        return ""
    return str(Path(*parts[strip_components:]))


def _safe_extract_zip(zip_path: Path, out_dir: Path, strip_components: int = 0, members: Optional[Iterable[str]] = None) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if members:
            names = [n for n in names if n in set(members)]
        for name in names:
            target = _strip_components(name, strip_components)
            if not target:
                continue
            target_path = out_dir / target
            if name.endswith("/"):
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(target_path, "wb") as dst:
                shutil.copyfileobj(src, dst)


def _safe_extract_tar(tar_path: Path, out_dir: Path, strip_components: int = 0, members: Optional[Iterable[str]] = None) -> None:
    with tarfile.open(tar_path) as tf:
        names = tf.getnames()
        if members:
            names = [n for n in names if n in set(members)]
        for name in names:
            target = _strip_components(name, strip_components)
            if not target:
                continue
            member = tf.getmember(name)
            if member.isdir():
                (out_dir / target).mkdir(parents=True, exist_ok=True)
                continue
            member.name = target
            tf.extract(member, path=out_dir)


def extract_archive(archive_path: str | Path, out_dir: str | Path, strip_components: int = 0, members: Optional[Iterable[str]] = None) -> None:
    archive_path = Path(archive_path)
    out_dir = ensure_dir(out_dir)
    suffix = "".join(archive_path.suffixes).lower()
    if suffix.endswith(".zip"):
        _safe_extract_zip(archive_path, out_dir, strip_components=strip_components, members=members)
        return
    if suffix.endswith(".tar") or suffix.endswith(".tar.gz") or suffix.endswith(".tgz") or suffix.endswith(".tar.bz2") or suffix.endswith(".tar.xz"):
        _safe_extract_tar(archive_path, out_dir, strip_components=strip_components, members=members)
        return
    raise ValueError(f"Unsupported archive format: {archive_path}")


def stage_download(
    src: str,
    cache_dir: Path,
    filename: Optional[str] = None,
    sha256: Optional[str] = None,
) -> Path:
    cache_dir = ensure_dir(cache_dir)
    if filename:
        cached = cache_dir / filename
        if cached.exists() and cached.stat().st_size > 0:
            if sha256:
                digest = sha256sum(cached)
                if digest.lower() == sha256.lower():
                    return cached
            else:
                return cached
    local_path = _resolve_source_to_file(src, cache_dir, filename=filename)
    if sha256:
        digest = sha256sum(local_path)
        if digest.lower() != sha256.lower():
            raise ValueError(f"SHA256 mismatch for {local_path}: expected {sha256}, got {digest}")
    return local_path
