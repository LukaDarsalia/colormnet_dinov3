from __future__ import annotations

import json
import os
import time
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Iterable, Union

import boto3
try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


_DEFAULT_IGNORE = {".DS_Store", "__pycache__"}


def require_s3_cfg(cfg: Dict[str, Any], stage: str) -> Dict[str, Any]:
    s3_cfg = cfg.get("s3")
    if not s3_cfg or not s3_cfg.get("enabled"):
        raise ValueError(f"s3.enabled must be true for stage '{stage}'.")
    if not s3_cfg.get("bucket"):
        raise ValueError(f"s3.bucket is required for stage '{stage}'.")
    return s3_cfg


def get_s3_client(s3_cfg: Dict[str, Any]):
    profile = s3_cfg.get("profile")
    region = s3_cfg.get("region")
    endpoint_url = s3_cfg.get("endpoint_url")
    session = boto3.session.Session(profile_name=profile) if profile else boto3.session.Session()
    return session.client("s3", region_name=region, endpoint_url=endpoint_url)


def build_stage_prefix(s3_cfg: Dict[str, Any], stage: str, run) -> str:
    base_prefix = s3_cfg.get("prefix", "colormnet")
    run_id = getattr(run, "id", None) or "run"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{base_prefix}/{stage}/{run_id}-{timestamp}"


def _extra_args(s3_cfg: Dict[str, Any]) -> Dict[str, Any]:
    extra: Dict[str, Any] = {}
    storage_class = s3_cfg.get("storage_class")
    if storage_class:
        extra["StorageClass"] = storage_class
    return extra


def _progress_bar(total: Optional[int], desc: str):
    if tqdm is None or total is None:
        return None, None
    pbar = tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=desc)

    def _callback(bytes_amount: int) -> None:
        pbar.update(bytes_amount)

    return pbar, _callback


def upload_file(client, bucket: str, local_path: str, s3_key: str, s3_cfg: Dict[str, Any]) -> None:
    extra = _extra_args(s3_cfg)
    progress = bool(s3_cfg.get("progress", True))
    min_bytes = int(s3_cfg.get("progress_min_bytes", 0) or 0)
    pbar = None
    callback = None
    if progress:
        try:
            size = os.path.getsize(local_path)
        except OSError:
            size = None
        if size is not None and size >= min_bytes:
            pbar, callback = _progress_bar(size, f"upload {Path(s3_key).name}")
    if extra:
        client.upload_file(local_path, bucket, s3_key, ExtraArgs=extra, Callback=callback)
    else:
        client.upload_file(local_path, bucket, s3_key, Callback=callback)
    if pbar is not None:
        pbar.close()


def upload_dir(client, bucket: str, local_dir: str, s3_prefix: str, s3_cfg: Dict[str, Any], ignore: Optional[set] = None) -> int:
    ignore = ignore or _DEFAULT_IGNORE
    count = 0
    for root, dirs, files in os.walk(local_dir):
        dirs[:] = [d for d in dirs if d not in ignore]
        for fname in files:
            if fname in ignore:
                continue
            local_path = os.path.join(root, fname)
            rel_path = os.path.relpath(local_path, local_dir)
            key = f"{s3_prefix}/{rel_path.replace(os.sep, '/') }"
            upload_file(client, bucket, local_path, key, s3_cfg)
            count += 1
    return count


def upload_json(client, bucket: str, s3_key: str, payload: Dict[str, Any], s3_cfg: Dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
        json.dump(payload, tmp, indent=2)
        tmp_path = tmp.name
    try:
        upload_file(client, bucket, tmp_path, s3_key, s3_cfg)
    finally:
        os.unlink(tmp_path)


def write_json_temp(payload: Dict[str, Any]) -> str:
    tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json")
    json.dump(payload, tmp, indent=2)
    tmp.flush()
    tmp.close()
    return tmp.name


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    path = uri[5:]
    bucket, _, key = path.partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return bucket, key


def download_file(client, bucket: str, s3_key: str, local_path: str, s3_cfg: Optional[Dict[str, Any]] = None) -> None:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3_cfg = s3_cfg or {}
    progress = bool(s3_cfg.get("progress", True))
    min_bytes = int(s3_cfg.get("progress_min_bytes", 0) or 0)
    pbar = None
    callback = None
    if progress:
        try:
            size = client.head_object(Bucket=bucket, Key=s3_key).get("ContentLength")
        except Exception:
            size = None
        if size is not None and size >= min_bytes:
            pbar, callback = _progress_bar(size, f"download {Path(s3_key).name}")
    client.download_file(bucket, s3_key, local_path, Callback=callback)
    if pbar is not None:
        pbar.close()


def _strip_components(path: str, strip_components: int) -> str:
    parts = path.split("/")
    if len(parts) <= strip_components:
        return ""
    return "/".join(parts[strip_components:])


def _detect_strip_components(names: Iterable[str]) -> int:
    top = set()
    for name in names:
        if not name or name.endswith("/"):
            continue
        top.add(name.split("/", 1)[0])
    return 1 if len(top) == 1 else 0


def extract_tar(archive_path: str, dest_dir: str, strip_components: Optional[int] = None) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as tf:
        names = tf.getnames()
        if strip_components is None:
            strip_components = _detect_strip_components(names)
        for name in names:
            target = _strip_components(name, strip_components)
            if not target:
                continue
            member = tf.getmember(name)
            if member.isdir():
                os.makedirs(os.path.join(dest_dir, target), exist_ok=True)
                continue
            member.name = target
            tf.extract(member, path=dest_dir)


def download_s3_archive_to_dir(uri: str, dest_dir: str, s3_cfg: Optional[Dict[str, Any]] = None) -> None:
    s3_cfg = s3_cfg or {}
    client = get_s3_client(s3_cfg)
    bucket, key = parse_s3_uri(uri)
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        download_file(client, bucket, key, tmp_path, s3_cfg=s3_cfg)
        extract_tar(tmp_path, dest_dir, strip_components=None)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _load_env() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        return
    load_dotenv(".env")


def generate_folder_name() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


class S3DataLoader:
    """Utility class for S3 upload/download operations with tar.gz compression."""

    def __init__(
        self,
        bucket: str,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        *,
        region: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        profile: Optional[str] = None,
        storage_class: Optional[str] = None,
        progress: bool = True,
        progress_min_bytes: int = 0,
    ):
        self.bucket = bucket
        self.storage_class = storage_class
        self.progress = progress
        self.progress_min_bytes = progress_min_bytes
        if profile:
            session = boto3.session.Session(profile_name=profile)
            self.s3_client = session.client("s3", region_name=region, endpoint_url=endpoint_url)
        elif access_key and secret_key:
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=region,
                endpoint_url=endpoint_url,
            )
        else:
            session = boto3.session.Session()
            self.s3_client = session.client("s3", region_name=region, endpoint_url=endpoint_url)

    def _extra_args(self) -> Dict[str, Any]:
        extra: Dict[str, Any] = {}
        if self.storage_class:
            extra["StorageClass"] = self.storage_class
        return extra

    def upload_file(self, filepath: Union[str, Path], s3_key: Optional[str] = None) -> str:
        filepath = Path(filepath)
        s3_key = s3_key or str(filepath)
        extra = self._extra_args()
        pbar = None
        callback = None
        if self.progress:
            try:
                size = filepath.stat().st_size
            except OSError:
                size = None
            if size is not None and size >= self.progress_min_bytes:
                pbar, callback = _progress_bar(size, f"upload {Path(s3_key).name}")
        if extra:
            self.s3_client.upload_file(str(filepath), self.bucket, s3_key, ExtraArgs=extra, Callback=callback)
        else:
            self.s3_client.upload_file(str(filepath), self.bucket, s3_key, Callback=callback)
        if pbar is not None:
            pbar.close()
        return s3_key

    def download_file(self, s3_key: str, local_path: Union[str, Path]) -> Path:
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        pbar = None
        callback = None
        if self.progress:
            try:
                size = self.s3_client.head_object(Bucket=self.bucket, Key=s3_key).get("ContentLength")
            except Exception:
                size = None
            if size is not None and size >= self.progress_min_bytes:
                pbar, callback = _progress_bar(size, f"download {Path(s3_key).name}")
        self.s3_client.download_file(self.bucket, s3_key, str(local_path), Callback=callback)
        if pbar is not None:
            pbar.close()
        return local_path

    def upload_as_tarball(self, folder_path: Union[str, Path], s3_key: Optional[str] = None) -> str:
        folder_path = Path(folder_path)
        s3_key = s3_key or f"{folder_path.name}.tar.gz"
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        try:
            with tarfile.open(tmp_path, "w:gz") as tar:
                for file_path in folder_path.rglob("*"):
                    if not file_path.is_dir():
                        arcname = file_path.relative_to(folder_path.parent)
                        tar.add(file_path, arcname=arcname)
            self.upload_file(tmp_path, s3_key=s3_key)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        return s3_key

    def download_and_extract_tarball(self, s3_key: str, extract_to: Union[str, Path] = ".") -> Path:
        extract_to = Path(extract_to)
        extract_to.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        try:
            self.download_file(s3_key, tmp_path)
            extract_tar(tmp_path, str(extract_to), strip_components=None)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        return extract_to


def get_s3_loader(bucket_name: str, *, s3_cfg: Optional[Dict[str, Any]] = None) -> S3DataLoader:
    _load_env()
    s3_cfg = s3_cfg or {}
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    return S3DataLoader(
        bucket=bucket_name,
        access_key=access_key,
        secret_key=secret_key,
        region=s3_cfg.get("region"),
        endpoint_url=s3_cfg.get("endpoint_url"),
        profile=s3_cfg.get("profile"),
        storage_class=s3_cfg.get("storage_class"),
        progress=bool(s3_cfg.get("progress", True)),
        progress_min_bytes=int(s3_cfg.get("progress_min_bytes", 0) or 0),
    )
