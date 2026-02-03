from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, Union

from src.utils.s3 import download_s3_archive_to_dir

import wandb


def _write_json_temp(payload: Dict[str, Any]) -> str:
    tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json")
    json.dump(payload, tmp, indent=2)
    tmp.flush()
    tmp.close()
    return tmp.name


def _read_json(path: Union[str, Path]) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def init_wandb_run(wandb_cfg: Optional[Dict[str, Any]], *, default_project: str, job_type: str, config: Optional[Dict[str, Any]] = None):
    cfg = wandb_cfg or {}
    kwargs: Dict[str, Any] = {
        "project": cfg.get("project", default_project),
        "job_type": cfg.get("job_type", job_type),
    }
    if cfg.get("entity"):
        kwargs["entity"] = cfg["entity"]
    if cfg.get("run_name"):
        kwargs["name"] = cfg["run_name"]
    if cfg.get("dir"):
        kwargs["dir"] = cfg["dir"]
    if cfg.get("tags"):
        kwargs["tags"] = cfg["tags"]
    if cfg.get("notes"):
        kwargs["notes"] = cfg["notes"]
    if config is not None:
        kwargs["config"] = config
    return wandb.init(**kwargs)


def create_dataset_artifact(
    name: str,
    dataset_dirs: Dict[str, str],
    *,
    artifact_type: str = "dataset",
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    manifest: Optional[Dict[str, Any]] = None,
    store_data: bool = True,
) -> wandb.Artifact:
    artifact = wandb.Artifact(name=name, type=artifact_type, description=description, metadata=metadata)
    manifest_payload = manifest or {}
    manifest_path = _write_json_temp(manifest_payload)
    artifact.add_file(manifest_path, name="manifest.json")

    if store_data:
        for key, dir_path in dataset_dirs.items():
            if dir_path is None:
                continue
            artifact.add_dir(dir_path, name=key)
    return artifact


def resolve_dataset_artifact(run, artifact_ref: str) -> Tuple[wandb.Artifact, str, Dict[str, Any]]:
    artifact = run.use_artifact(artifact_ref)
    root = artifact.download()
    manifest_path = Path(root) / "manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        s3_path = Path(root) / "s3.json"
        if s3_path.exists():
            try:
                s3_info = _read_json(s3_path)
            except json.JSONDecodeError:
                s3_info = {}
            datasets = (s3_info or {}).get("datasets", {})
            for key in manifest.values():
                if not key:
                    continue
                local_dir = Path(root) / key
                if local_dir.exists():
                    continue
                s3_meta = datasets.get(key)
                if not s3_meta:
                    continue
                uri = s3_meta.get("uri")
                if not uri:
                    continue
                download_s3_archive_to_dir(uri, local_dir)
        return artifact, root, manifest
    metadata = artifact.metadata or {}
    upstream = metadata.get("input_artifact")
    if upstream:
        return resolve_dataset_artifact(run, upstream)
    raise FileNotFoundError(f"manifest.json not found in artifact {artifact_ref}")


def resolve_split_artifact(run, artifact_ref: str) -> Tuple[wandb.Artifact, Dict[str, Any]]:
    artifact = run.use_artifact(artifact_ref)
    root = artifact.download()
    split_path = Path(root) / "split.json"
    if not split_path.exists():
        raise FileNotFoundError(f"split.json not found in artifact {artifact_ref}")
    return artifact, _read_json(split_path)
