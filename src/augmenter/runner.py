from __future__ import annotations

import argparse
import os
import tempfile

import wandb

from src.utils.artifacts import init_wandb_run, resolve_dataset_artifact
from src.utils.config import load_config
from src.utils.s3 import build_stage_prefix, get_s3_client, require_s3_cfg, upload_dir, upload_json, write_json_temp


DEFAULT_PROJECT = "ColorMNet"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-artifact", default=None, help="Override input artifact reference")
    args = parser.parse_args()

    cfg = load_config(args.config)
    wandb_cfg = cfg.get("wandb", {})
    run = init_wandb_run(wandb_cfg, default_project=DEFAULT_PROJECT, job_type="augmenter", config=cfg)

    input_artifact = args.input_artifact or cfg.get("input_artifact")
    if not input_artifact:
        raise ValueError("input_artifact is required for augmenter stage")

    output_cfg = cfg.get("output_artifact", {})
    name = output_cfg.get("name", "colormnet-dataset-aug")
    description = output_cfg.get("description") or "Pass-through artifact. Augmentation happens online in dataset pipeline."

    augmentations = [
        "ColorJitter(0.01, 0.01, 0.01, 0)",
        "RandomAffine(degrees=15, shear=10, fill=im_mean)",
        "ColorJitter(0.1, 0.03, 0.03, 0)",
        "RandomHorizontalFlip()",
        "RandomResizedCrop((448, 448), scale=(0.36, 1.00))",
    ]

    artifact = wandb.Artifact(
        name=name,
        type=output_cfg.get("type", "dataset"),
        description=description,
        metadata={
            "input_artifact": input_artifact,
            "passthrough": True,
            "augmentation_source": "shared.dataset.vos_dataset.DAVISVidevoDataset",
            "augmentations": augmentations,
        },
    )

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
        tmp.write("ColorMNet uses online augmentation in the dataset loader. This artifact is a pass-through.\n")
        note_path = tmp.name
    artifact.add_file(note_path, name="README.txt")

    s3_cfg = require_s3_cfg(cfg, "augmenter")
    client = get_s3_client(s3_cfg)
    stage_prefix = build_stage_prefix(s3_cfg, "augmenter", run)
    bucket = s3_cfg["bucket"]

    s3_info = {
        "stage": "augmenter",
        "prefix": stage_prefix,
        "passthrough": True,
    }

    upstream = run.use_artifact(input_artifact)
    upstream_s3 = (upstream.metadata or {}).get("s3")
    if upstream_s3 and s3_cfg.get("reuse_upstream", True):
        s3_info["upstream"] = upstream_s3
        manifest_key = f"{stage_prefix}/manifest.json"
        upload_json(client, bucket, manifest_key, s3_info, s3_cfg)
        s3_info["manifest_uri"] = f"s3://{bucket}/{manifest_key}"
    else:
        _, dataset_root, manifest = resolve_dataset_artifact(run, input_artifact)
        s3_info["datasets"] = {}
        for key, subdir in manifest.items():
            dir_path = os.path.join(dataset_root, subdir)
            target_prefix = f"{stage_prefix}/datasets/{key}"
            file_count = upload_dir(client, bucket, dir_path, target_prefix, s3_cfg)
            s3_info["datasets"][key] = {
                "uri": f"s3://{bucket}/{target_prefix}",
                "files": file_count,
            }
        manifest_key = f"{stage_prefix}/manifest.json"
        upload_json(client, bucket, manifest_key, manifest, s3_cfg)
        s3_info["manifest_uri"] = f"s3://{bucket}/{manifest_key}"

    s3_info_path = write_json_temp(s3_info)
    artifact.add_file(s3_info_path, name="s3.json")
    os.unlink(s3_info_path)
    artifact.metadata = {**(artifact.metadata or {}), "s3": s3_info}

    run.log_artifact(artifact)
    run.summary["s3/augmenter"] = s3_info
    run.finish()


if __name__ == "__main__":
    main()
