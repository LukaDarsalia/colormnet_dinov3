from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from typing import List

import wandb

from src.utils.artifacts import init_wandb_run, resolve_dataset_artifact
from src.utils.config import load_config
from src.utils.s3 import build_stage_prefix, get_s3_client, require_s3_cfg, upload_file, write_json_temp


DEFAULT_PROJECT = "ColorMNet"


def _list_video_dirs(root: str) -> List[str]:
    return [
        d for d in sorted(os.listdir(root))
        if not d.startswith('.') and os.path.isdir(os.path.join(root, d))
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-artifact", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    wandb_cfg = cfg.get("wandb", {})
    run = init_wandb_run(wandb_cfg, default_project=DEFAULT_PROJECT, job_type="splitter", config=cfg)

    input_artifact = args.input_artifact or cfg.get("input_artifact")
    if not input_artifact:
        raise ValueError("input_artifact is required for splitter stage")

    _, dataset_root, manifest = resolve_dataset_artifact(run, input_artifact)

    split_cfg = cfg.get("split", {})
    mode = split_cfg.get("mode", "auto")
    val_fraction = float(split_cfg.get("val_fraction", 0.1))
    seed = int(split_cfg.get("seed", 1337))
    shuffle = bool(split_cfg.get("shuffle", True))
    max_videos = split_cfg.get("max_videos")

    train_key = manifest.get("train_root")
    val_key = manifest.get("val_root")
    if not train_key:
        raise ValueError("manifest.train_root is required for splitter stage")

    train_root = os.path.join(dataset_root, train_key)
    val_root = os.path.join(dataset_root, val_key) if val_key else None

    if (mode in ("auto", "use_existing")) and val_root and os.path.isdir(val_root):
        train_videos = _list_video_dirs(train_root)
        val_videos = _list_video_dirs(val_root)
    else:
        videos = _list_video_dirs(train_root)
        if max_videos:
            videos = videos[: int(max_videos)]
        if shuffle:
            random.Random(seed).shuffle(videos)
        if val_fraction <= 0:
            train_videos, val_videos = videos, []
        else:
            split_idx = max(1, int(len(videos) * (1 - val_fraction)))
            train_videos, val_videos = videos[:split_idx], videos[split_idx:]

    split_payload = {
        "dataset_artifact": input_artifact,
        "mode": mode,
        "seed": seed,
        "val_fraction": val_fraction,
        "train_videos": train_videos,
        "val_videos": val_videos,
    }

    output_cfg = cfg.get("output_artifact", {})
    name = output_cfg.get("name", "colormnet-split")
    description = output_cfg.get("description") or "Train/val split lists for ColorMNet."

    artifact = wandb.Artifact(
        name=name,
        type=output_cfg.get("type", "split"),
        description=description,
        metadata={
            "dataset_artifact": input_artifact,
            "train_count": len(train_videos),
            "val_count": len(val_videos),
        },
    )

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
        json.dump(split_payload, tmp, indent=2)
        split_path = tmp.name

    artifact.add_file(split_path, name="split.json")

    s3_cfg = require_s3_cfg(cfg, "splitter")
    client = get_s3_client(s3_cfg)
    stage_prefix = build_stage_prefix(s3_cfg, "splitter", run)
    bucket = s3_cfg["bucket"]

    split_key = f"{stage_prefix}/split.json"
    upload_file(client, bucket, split_path, split_key, s3_cfg)
    s3_info = {
        "stage": "splitter",
        "prefix": stage_prefix,
        "split_uri": f"s3://{bucket}/{split_key}",
        "dataset_artifact": input_artifact,
        "train_count": len(train_videos),
        "val_count": len(val_videos),
    }
    s3_info_path = write_json_temp(s3_info)
    artifact.add_file(s3_info_path, name="s3.json")
    os.unlink(s3_info_path)
    artifact.metadata = {**(artifact.metadata or {}), "s3": s3_info}

    run.log({"split/train_count": len(train_videos), "split/val_count": len(val_videos)})
    run.log_artifact(artifact)
    run.summary["s3/splitter"] = s3_info
    run.finish()


if __name__ == "__main__":
    main()
