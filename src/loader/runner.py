from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
import os
import tarfile
import tempfile

import wandb

from src.utils.artifacts import create_dataset_artifact, init_wandb_run
from src.utils.config import load_config
from src.utils.samples import sample_images
from src.utils.s3 import build_stage_prefix, get_s3_client, require_s3_cfg, upload_dir, upload_json, upload_file, write_json_temp

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_PROJECT = "ColorMNet"


def _resolve_path(p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    return str(Path(p).expanduser().resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    wandb_cfg = cfg.get("wandb", {})
    run = init_wandb_run(wandb_cfg, default_project=DEFAULT_PROJECT, job_type="loader", config=cfg)

    data_cfg = cfg.get("data", {})
    train_root = _resolve_path(data_cfg.get("train_root"))
    val_root = _resolve_path(data_cfg.get("val_root"))
    eval_input_root = _resolve_path(data_cfg.get("eval_input_root"))
    eval_ref_root = _resolve_path(data_cfg.get("eval_ref_root"))

    if not train_root:
        raise ValueError("data.train_root is required for loader stage")

    dataset_dirs = {}
    manifest = {}
    if train_root:
        dataset_dirs["train"] = train_root
        manifest["train_root"] = "train"
    if val_root:
        dataset_dirs["val"] = val_root
        manifest["val_root"] = "val"
    if eval_input_root:
        dataset_dirs["eval_input"] = eval_input_root
        manifest["eval_input_root"] = "eval_input"
    if eval_ref_root:
        dataset_dirs["eval_ref"] = eval_ref_root
        manifest["eval_ref_root"] = "eval_ref"

    artifact_cfg = cfg.get("artifact", {})
    name = artifact_cfg.get("name", "colormnet-dataset")
    description = artifact_cfg.get("description")
    store_data = bool(artifact_cfg.get("store_data", True))
    metadata = artifact_cfg.get("metadata") or {}
    metadata["manifest_keys"] = list(manifest.keys())

    artifact = create_dataset_artifact(
        name,
        dataset_dirs,
        artifact_type=artifact_cfg.get("type", "dataset"),
        description=description,
        metadata=metadata,
        manifest=manifest,
        store_data=store_data,
    )

    s3_info = None
    s3_cfg = require_s3_cfg(cfg, "loader")
    client = get_s3_client(s3_cfg)
    stage_prefix = build_stage_prefix(s3_cfg, "loader", run)
    bucket = s3_cfg["bucket"]
    s3_info = {
        "stage": "loader",
        "prefix": stage_prefix,
        "datasets": {},
    }
    pack_datasets = bool(s3_cfg.get("pack_datasets", False))
    pack_compression = s3_cfg.get("pack_compression", "gz")
    for key, dir_path in dataset_dirs.items():
        if not dir_path:
            continue
        target_prefix = f"{stage_prefix}/datasets/{key}"
        if pack_datasets:
            print(f"Packing dataset '{key}' for S3 upload...")
            with tempfile.NamedTemporaryFile(suffix=f".tar.{pack_compression}", delete=False) as tmp:
                archive_path = tmp.name
            mode = f"w:{pack_compression}" if pack_compression else "w"
            with tarfile.open(archive_path, mode) as tf:
                base_name = Path(dir_path).name
                file_paths = []
                for root, dirs, files in os.walk(dir_path):
                    dirs[:] = [d for d in dirs if d not in {".DS_Store", "__pycache__"}]
                    for fname in files:
                        if fname in {".DS_Store"}:
                            continue
                        file_paths.append(os.path.join(root, fname))
                iterator = file_paths
                if tqdm is not None:
                    iterator = tqdm(file_paths, desc=f"  adding {base_name}", unit="file")
                for fpath in iterator:
                    rel = os.path.relpath(fpath, dir_path)
                    arcname = os.path.join(base_name, rel)
                    tf.add(fpath, arcname=arcname)
            archive_key = f"{target_prefix}/{Path(dir_path).name}.tar.{pack_compression}" if pack_compression else f"{target_prefix}/{Path(dir_path).name}.tar"
            upload_file(client, bucket, archive_path, archive_key, s3_cfg)
            os.unlink(archive_path)
            s3_info["datasets"][key] = {
                "uri": f"s3://{bucket}/{archive_key}",
                "packed": True,
            }
            artifact.add_reference(f"s3://{bucket}/{archive_key}", name=key)
        else:
            file_count = upload_dir(client, bucket, dir_path, target_prefix, s3_cfg)
            s3_info["datasets"][key] = {
                "uri": f"s3://{bucket}/{target_prefix}",
                "files": file_count,
                "packed": False,
            }
            artifact.add_reference(f"s3://{bucket}/{target_prefix}", name=key)
    manifest_key = f"{stage_prefix}/manifest.json"
    upload_json(client, bucket, manifest_key, manifest, s3_cfg)
    s3_info["manifest_uri"] = f"s3://{bucket}/{manifest_key}"

    s3_info_path = write_json_temp(s3_info)
    artifact.add_file(s3_info_path, name="s3.json")
    os.unlink(s3_info_path)
    artifact.metadata = {**(artifact.metadata or {}), "s3": s3_info}
    run.log_artifact(artifact)
    run.summary["s3/loader"] = s3_info

    logging_cfg = cfg.get("logging", {})
    sample_videos = int(logging_cfg.get("sample_videos", 1))
    sample_frames = int(logging_cfg.get("sample_frames", 3))

    train_samples = sample_images(train_root, max_videos=sample_videos, max_frames=sample_frames)
    if train_samples:
        run.log({
            "samples/train": [wandb.Image(img, caption=cap) for img, cap in train_samples]
        })

    if eval_input_root:
        eval_samples = sample_images(eval_input_root, max_videos=sample_videos, max_frames=sample_frames)
        if eval_samples:
            run.log({
                "samples/eval_input": [wandb.Image(img, caption=cap) for img, cap in eval_samples]
            })

    run.finish()


if __name__ == "__main__":
    main()
