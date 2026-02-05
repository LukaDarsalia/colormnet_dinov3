from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import os
from pathlib import Path
import cv2
from types import SimpleNamespace
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

import wandb

from shared.evaluation_matrics.evaluation import calculate_psnr_for_folder
from shared.evaluation_matrics.fid import calculate_fid_yyx
from shared.evaluation_matrics.cdc import calculate_cdc_yyx
from shared.inference.data.test_datasets import DAVISTestDataset_221128_TransColorization_batch
from shared.inference.inference_core import InferenceCore
from shared.model.network import ColorMNet
from shared.util.transforms import lab2rgb_transform_PIL

from src.utils.artifacts import init_wandb_run, resolve_dataset_artifact
from src.utils.config import load_config
from src.utils.samples import sample_images
from src.utils.s3 import build_stage_prefix, get_s3_client, require_s3_cfg, upload_dir, upload_file, upload_json, write_json_temp


DEFAULT_PROJECT = "ColorMNet"


def _load_model(model_path: str, config: dict) -> ColorMNet:
    network = ColorMNet(config, model_path).cuda().eval()
    if model_path is not None:
        model_weights = torch.load(model_path)
        network.load_weights(model_weights, init_as_zero_if_needed=True)
    return network


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_output_frame(out_dir: str, vid_name: str, frame_name: str, rgb, prob):
    this_out_path = os.path.join(out_dir, vid_name)
    os.makedirs(this_out_path, exist_ok=True)

    out_mask_final = lab2rgb_transform_PIL(torch.cat([rgb[:1, :, :], prob], dim=0))
    out_mask_final = (out_mask_final * 255).astype(np.uint8)

    out_img = Image.fromarray(out_mask_final)
    out_img.save(os.path.join(this_out_path, f"{Path(frame_name).stem}.png"))


def _list_video_dirs(root: str) -> List[str]:
    return [
        d for d in sorted(os.listdir(root))
        if not d.startswith('.') and os.path.isdir(os.path.join(root, d))
    ]


def _write_videos_from_frames(frames_root: str, videos_root: str, fps: int = 24, max_videos: int | None = None) -> list[str]:
    os.makedirs(videos_root, exist_ok=True)
    outputs = []
    for vid in _list_video_dirs(frames_root):
        vid_dir = os.path.join(frames_root, vid)
        frames = sorted([f for f in os.listdir(vid_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not frames:
            continue
        first = cv2.imread(os.path.join(vid_dir, frames[0]))
        if first is None:
            continue
        height, width = first.shape[:2]
        out_path = os.path.join(videos_root, f"{vid}.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        for frame in frames:
            img = cv2.imread(os.path.join(vid_dir, frame))
            if img is None:
                continue
            writer.write(img)
        writer.release()
        outputs.append(out_path)
        if max_videos is not None and len(outputs) >= max_videos:
            break
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-artifact", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    wandb_cfg = cfg.get("wandb", {})
    run = init_wandb_run(wandb_cfg, default_project=DEFAULT_PROJECT, job_type="evaluator", config=cfg)

    input_artifact = args.input_artifact or cfg.get("input_artifact")
    if not input_artifact:
        raise ValueError("input_artifact is required for evaluator stage")

    _, dataset_root, manifest = resolve_dataset_artifact(run, input_artifact)

    eval_input_key = manifest.get("eval_input_root")
    eval_ref_key = manifest.get("eval_ref_root")
    if not eval_input_key:
        raise ValueError("Dataset manifest missing eval_input_root for evaluation")

    eval_input_root = os.path.join(dataset_root, eval_input_key)
    eval_ref_root = os.path.join(dataset_root, eval_ref_key) if eval_ref_key else None

    model_cfg = cfg.get("model", {})
    model_path = model_cfg.get("path")
    model_artifact = model_cfg.get("artifact")

    if model_artifact:
        artifact = run.use_artifact(model_artifact)
        model_root = artifact.download()
        if model_cfg.get("file_name"):
            model_path = os.path.join(model_root, model_cfg["file_name"])
        else:
            candidates = [p for p in Path(model_root).rglob("*.pth")]
            if not candidates:
                raise FileNotFoundError(f"No .pth files found in {model_root}")
            model_path = str(candidates[0])

    if not model_path:
        raise ValueError("model.path or model.artifact is required for evaluation")

    inference_cfg = cfg.get("inference", {})
    size = int(inference_cfg.get("size", -1))
    reverse = bool(inference_cfg.get("reverse", False))
    first_frame_not_exemplar = bool(inference_cfg.get("FirstFrameIsNotExemplar", False))
    exemplar_path = inference_cfg.get("exemplar_path")
    if exemplar_path:
        exemplar_path = str(Path(exemplar_path).expanduser())
        if not os.path.isabs(exemplar_path):
            exemplar_path = str((_REPO_ROOT / exemplar_path).resolve())
        if not Path(exemplar_path).is_file():
            raise FileNotFoundError(f"Exemplar image not found: {exemplar_path}")
        if not first_frame_not_exemplar:
            first_frame_not_exemplar = True

    config = {
        "enable_long_term": not bool(inference_cfg.get("disable_long_term", False)),
        "max_mid_term_frames": int(inference_cfg.get("max_mid_term_frames", 10)),
        "min_mid_term_frames": int(inference_cfg.get("min_mid_term_frames", 5)),
        "max_long_term_elements": int(inference_cfg.get("max_long_term_elements", 10000)),
        "num_prototypes": int(inference_cfg.get("num_prototypes", 128)),
        "top_k": int(inference_cfg.get("top_k", 30)),
        "mem_every": int(inference_cfg.get("mem_every", 5)),
        "deep_update_every": int(inference_cfg.get("deep_update_every", -1)),
        "benchmark": bool(inference_cfg.get("benchmark", False)),
        "flip": bool(inference_cfg.get("flip", False)),
        "FirstFrameIsNotExemplar": first_frame_not_exemplar,
    }
    config["dino"] = cfg.get("dino", {})

    output_cfg = cfg.get("output", {})
    output_root = output_cfg.get("dir", "./outputs/eval")
    output_root = str(Path(output_root).expanduser().resolve())
    _ensure_dir(output_root)

    torch.autograd.set_grad_enabled(False)

    args_ns = SimpleNamespace(reverse=reverse)
    meta_dataset = DAVISTestDataset_221128_TransColorization_batch(
        eval_input_root,
        imset=eval_ref_root or eval_input_root,
        size=size,
        args=args_ns,
        exemplar_path=exemplar_path,
    )

    network = _load_model(model_path, config)

    meta_loader = meta_dataset.get_datasets()
    total_videos = len(meta_dataset)

    for vid_idx, vid_reader in enumerate(meta_loader, start=1):
        loader = DataLoader(vid_reader, batch_size=1, shuffle=False, num_workers=2)
        vid_name = vid_reader.vid_name
        vid_length = len(loader)
        print(f"[eval] Video {vid_idx}/{total_videos}: {vid_name} ({vid_length} frames)", flush=True)
        config["enable_long_term_count_usage"] = (
            config["enable_long_term"] and
            (vid_length / (config["max_mid_term_frames"] - config["min_mid_term_frames"]) * config["num_prototypes"])
            >= config["max_long_term_elements"]
        )

        processor = InferenceCore(network, config=config)
        first_mask_loaded = False

        for ti, data in enumerate(loader):
            with torch.cuda.amp.autocast(enabled=not config["benchmark"]):
                rgb = data["rgb"].cuda()[0]
                msk = data.get("mask")
                if not config["FirstFrameIsNotExemplar"]:
                    msk = msk[:, 1:3, :, :] if msk is not None else None

                info = data["info"]
                frame = info["frame"][0]
                shape = info["shape"]
                need_resize = info["need_resize"][0]

                if not first_mask_loaded:
                    if msk is not None:
                        first_mask_loaded = True
                    else:
                        continue

                if config["flip"]:
                    rgb = torch.flip(rgb, dims=[-1])
                    msk = torch.flip(msk, dims=[-1]) if msk is not None else None

                if msk is not None:
                    msk = torch.Tensor(msk[0]).cuda()
                    if need_resize:
                        msk = vid_reader.resize_mask(msk.unsqueeze(0))[0]
                    processor.set_all_labels(list(range(1, 3)))
                    labels = range(1, 3)
                else:
                    labels = None

                if config["FirstFrameIsNotExemplar"]:
                    prob = processor.step_AnyExemplar(
                        rgb,
                        msk[:1, :, :].repeat(3, 1, 1) if msk is not None else None,
                        msk[1:3, :, :] if msk is not None else None,
                        labels,
                        end=(ti == vid_length - 1),
                    )
                else:
                    prob = processor.step(rgb, msk, labels, end=(ti == vid_length - 1))

                if need_resize:
                    prob = F.interpolate(prob.unsqueeze(1), shape, mode="bilinear", align_corners=False)[:, 0]

                if config["flip"]:
                    prob = torch.flip(prob, dims=[-1])

                _save_output_frame(output_root, vid_name, frame, rgb, prob)
            if (ti + 1) % 10 == 0 or (ti + 1) == vid_length:
                print(f"[eval] {vid_name}: {ti + 1}/{vid_length} frames", flush=True)

    metrics_cfg = cfg.get("metrics", {})
    metrics = {}

    if eval_ref_root:
        if metrics_cfg.get("lpips"):
            try:
                import lpips  # type: ignore
            except ImportError as exc:
                raise RuntimeError("lpips is required for LPIPS metric. Install with `pip install lpips`.") from exc
            loss_fn_alex = lpips.LPIPS(net="alex", verbose=False)
        else:
            loss_fn_alex = None

        psnr, ssim, colorfulness, lpips_score = calculate_psnr_for_folder(
            eval_ref_root,
            output_root,
            loss_fn_alex,
            bool(metrics_cfg.get("psnr", True)),
            bool(metrics_cfg.get("ssim", True)),
            bool(metrics_cfg.get("lpips", False)),
            bool(metrics_cfg.get("colorfulness", False)),
        )

        if metrics_cfg.get("psnr", True):
            metrics["eval/psnr"] = psnr
        if metrics_cfg.get("ssim", True):
            metrics["eval/ssim"] = ssim
        if metrics_cfg.get("colorfulness", False):
            metrics["eval/colorfulness"] = colorfulness
        if metrics_cfg.get("lpips", False):
            metrics["eval/lpips"] = lpips_score

        if metrics_cfg.get("fid", False):
            metrics["eval/fid"] = calculate_fid_yyx(output_root, eval_ref_root, 100)
        if metrics_cfg.get("cdc", False):
            metrics["eval/cdc"] = calculate_cdc_yyx(output_root)

    if metrics:
        run.log(metrics)

    logging_cfg = cfg.get("logging", {})
    sample_videos = int(logging_cfg.get("sample_videos", 1))
    sample_frames = int(logging_cfg.get("sample_frames", 3))

    pred_samples = sample_images(output_root, max_videos=sample_videos, max_frames=sample_frames)
    if pred_samples:
        run.log({
            "samples/pred": [wandb.Image(img, caption=cap) for img, cap in pred_samples]
        })

    if eval_ref_root:
        gt_samples = sample_images(eval_ref_root, max_videos=sample_videos, max_frames=sample_frames)
        if gt_samples:
            run.log({
                "samples/gt": [wandb.Image(img, caption=cap) for img, cap in gt_samples]
            })

    output_cfg = cfg.get("output", {})
    create_videos = bool(output_cfg.get("create_videos", False))
    video_fps = int(output_cfg.get("video_fps", 24))
    videos_dir = None
    video_paths = []
    if create_videos or sample_videos > 0:
        videos_dir = os.path.join(output_root, "_videos")
        max_videos = None if create_videos else sample_videos
        video_paths = _write_videos_from_frames(output_root, videos_dir, fps=video_fps, max_videos=max_videos)
        if sample_videos > 0 and video_paths:
            run.log({
                "samples/video": [
                    wandb.Video(path, fps=video_fps, format="mp4")
                    for path in video_paths[:sample_videos]
                ]
            })

    s3_cfg = require_s3_cfg(cfg, "evaluator")
    client = get_s3_client(s3_cfg)
    stage_prefix = build_stage_prefix(s3_cfg, "evaluator", run)
    bucket = s3_cfg["bucket"]
    s3_info = {
        "stage": "evaluator",
        "prefix": stage_prefix,
        "outputs": {},
    }
    outputs_prefix = f"{stage_prefix}/outputs"
    ignore = {"_videos"} if videos_dir else None
    outputs_count = upload_dir(client, bucket, output_root, outputs_prefix, s3_cfg, ignore=ignore)
    s3_info["outputs"]["frames_uri"] = f"s3://{bucket}/{outputs_prefix}"
    s3_info["outputs"]["frames_files"] = outputs_count

    videos_dir_for_upload = videos_dir if create_videos else None
    if videos_dir_for_upload:
        videos_prefix = f"{stage_prefix}/videos"
        videos_count = upload_dir(client, bucket, videos_dir_for_upload, videos_prefix, s3_cfg)
        s3_info["outputs"]["videos_uri"] = f"s3://{bucket}/{videos_prefix}"
        s3_info["outputs"]["videos_files"] = videos_count

    if metrics:
        metrics_key = f"{stage_prefix}/metrics.json"
        upload_json(client, bucket, metrics_key, metrics, s3_cfg)
        s3_info["metrics_uri"] = f"s3://{bucket}/{metrics_key}"

    s3_info_path = write_json_temp(s3_info)
    run.save(s3_info_path, base_path=os.path.dirname(s3_info_path))
    os.unlink(s3_info_path)
    run.summary["s3/evaluator"] = s3_info

    output_artifact_cfg = cfg.get("output_artifact", {})
    if output_artifact_cfg.get("enabled"):
        art = wandb.Artifact(
            name=output_artifact_cfg.get("name", "colormnet-eval-output"),
            type=output_artifact_cfg.get("type", "predictions"),
            description=output_artifact_cfg.get("description", "Evaluation outputs"),
        )
        art.add_dir(output_root)
        run.log_artifact(art)

    run.finish()


if __name__ == "__main__":
    main()
