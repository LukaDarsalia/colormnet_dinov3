from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import math
import os
import random
import socket
from os import path
from pathlib import Path

import git
import numpy as np
import torch
import torch.distributed as distributed
from torch.utils.data import DataLoader, ConcatDataset

import wandb

from shared.dataset.vos_dataset import DAVISVidevoDataset
from shared.inference.data.test_datasets import DAVISTestDataset_221128_TransColorization_batch
from shared.model.trainer import ColorMNetTrainer
from shared.util.logger import TensorboardLogger

from src.utils.artifacts import resolve_dataset_artifact, resolve_split_artifact
from src.utils.config import load_config
from src.utils.s3 import build_stage_prefix, get_s3_client, require_s3_cfg, upload_file, write_json_temp


DEFAULT_PROJECT = "ColorMNet"


def _build_base_config(cfg: dict, data_paths: dict) -> dict:
    training = cfg.get("training", {})
    base = {
        "benchmark": bool(training.get("benchmark", False)),
        "no_amp": bool(training.get("no_amp", False)),
        "davis_root": data_paths["train_root"],
        "validation_root": data_paths["val_root"],
        "num_workers": int(training.get("num_workers", 16)),
        "key_dim": int(training.get("key_dim", 64)),
        "value_dim": int(training.get("value_dim", 512)),
        "hidden_dim": int(training.get("hidden_dim", 64)),
        "deep_update_prob": float(training.get("deep_update_prob", 0.2)),
        "stages": str(training.get("stages", "2")),
        "savepath": training.get("savepath", "./wandb_save_dir"),
        "gamma": float(training.get("gamma", 0.1)),
        "weight_decay": float(training.get("weight_decay", 0.05)),
        "load_network": training.get("load_network"),
        "load_checkpoint": training.get("load_checkpoint"),
        "log_text_interval": int(training.get("log_text_interval", 100)),
        "log_image_interval": int(training.get("log_image_interval", 100)),
        "save_network_interval": int(training.get("save_network_interval", 2500)),
        "save_checkpoint_interval": int(training.get("save_checkpoint_interval", 5000)),
        "exp_id": training.get("exp_id") or "NULL",
        "debug": bool(training.get("debug", False)),
    }
    base["dino"] = cfg.get("dino", {})
    base["amp"] = not base["no_amp"]
    return base


def _get_stage_params(training: dict, stage: str) -> dict:
    stage_cfg = training.get(f"stage{stage}")
    if not stage_cfg:
        raise ValueError(f"Missing training.stage{stage} config")
    return {
        "batch_size": int(stage_cfg.get("batch_size")),
        "iterations": int(stage_cfg.get("iterations")),
        "finetune": int(stage_cfg.get("finetune", 0)),
        "steps": list(stage_cfg.get("steps", [])),
        "lr": float(stage_cfg.get("lr")),
        "num_ref_frames": int(stage_cfg.get("num_ref_frames")),
        "num_frames": int(stage_cfg.get("num_frames")),
        "start_warm": int(stage_cfg.get("start_warm", 0)),
        "end_warm": int(stage_cfg.get("end_warm", 0)),
    }


def _init_wandb_run(wandb_cfg: dict, config: dict):
    project = wandb_cfg.get("project", DEFAULT_PROJECT)
    entity = wandb_cfg.get("entity")
    job_type = wandb_cfg.get("job_type", "training")
    run_name = wandb_cfg.get("run_name") or config.get("exp_id")
    run_dir = wandb_cfg.get("dir", config.get("savepath"))
    tags = wandb_cfg.get("tags")

    kwargs = {
        "project": project,
        "job_type": job_type,
        "name": run_name,
        "dir": run_dir,
        "config": config,
        "notes": socket.gethostname(),
    }
    if entity:
        kwargs["entity"] = entity
    if tags:
        kwargs["tags"] = tags
    return wandb.init(**kwargs)


def _collect_checkpoints(save_dir: str):
    root = Path(save_dir)
    if not root.exists():
        return []
    return sorted([p for p in root.glob("*.pth")])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-artifact", default=None, help="Split artifact reference")
    args = parser.parse_args()

    cfg = load_config(args.config)
    wandb_cfg = cfg.get("wandb", {})
    s3_cfg = require_s3_cfg(cfg, "trainer")

    # Init distributed environment
    distributed.init_process_group(backend="nccl")
    print(f"CUDA Device count: {torch.cuda.device_count()}")

    local_rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    torch.cuda.set_device(local_rank)

    # Get current git info
    repo = git.Repo(".")
    git_info = f"{repo.active_branch} {repo.head.commit.hexsha}"

    input_artifact = args.input_artifact or cfg.get("input_artifact")
    if not input_artifact:
        raise ValueError("input_artifact is required for trainer stage")

    training_cfg = cfg.get("training", {})
    stages = str(training_cfg.get("stages", "2"))
    stages_to_perform = list(stages)

    base_config = None
    dataset_artifact_ref = None
    train_root = None
    val_root = None
    train_videos = []
    val_videos = []

    network_in_memory = None

    for si, stage in enumerate(stages_to_perform):
        # Set seed to ensure the same initialization
        torch.manual_seed(14159265)
        np.random.seed(14159265)
        random.seed(14159265)

        stage_config = _get_stage_params(training_cfg, stage)
        stage_exp_id = training_cfg.get("exp_id") or "NULL"
        if stage_exp_id != "NULL":
            stage_exp_id = f"{stage_exp_id}_s{stages[:si+1]}"
        wb_run = _init_wandb_run(wandb_cfg, {**training_cfg, **stage_config, "exp_id": stage_exp_id})

        if base_config is None:
            split_artifact, split_data = resolve_split_artifact(wb_run, input_artifact)
            dataset_artifact_ref = split_data["dataset_artifact"]
            _, dataset_root, manifest = resolve_dataset_artifact(wb_run, dataset_artifact_ref)

            train_key = manifest.get("train_root")
            val_key = manifest.get("val_root") or manifest.get("train_root")
            if not train_key:
                raise ValueError("Dataset manifest missing train_root")

            train_root = os.path.join(dataset_root, train_key)
            val_root = os.path.join(dataset_root, val_key)

            train_videos = split_data.get("train_videos", [])
            val_videos = split_data.get("val_videos", [])

            base_config = _build_base_config(cfg, {"train_root": train_root, "val_root": val_root})
            if base_config["benchmark"]:
                torch.backends.cudnn.benchmark = True
        else:
            wb_run.use_artifact(input_artifact)
            if dataset_artifact_ref:
                wb_run.use_artifact(dataset_artifact_ref)

        config = {**base_config, **stage_config}
        config["exp_id"] = stage_exp_id

        config["single_object"] = (stage == "0")
        config["num_gpus"] = world_size
        if config["batch_size"] // config["num_gpus"] * config["num_gpus"] != config["batch_size"]:
            raise ValueError("Batch size must be divisible by the number of GPUs.")
        config["batch_size"] //= config["num_gpus"]
        config["num_workers"] //= config["num_gpus"]
        print(f"We are assuming {config['num_gpus']} GPUs.")
        print(f"We are now starting stage {stage}")

        wb_run.config.update(config, allow_val_change=True)

        savepath = config["savepath"]
        if local_rank == 0:
            if config["exp_id"].lower() != "null":
                print("I will take the role of logging!")
                long_id = f"{config['exp_id']}"
            else:
                long_id = None
            logger = TensorboardLogger(config["exp_id"], long_id, git_info, False, savepath=savepath)
            logger.log_string("hyperpara", str(config))
            model = ColorMNetTrainer(
                config,
                logger=logger,
                save_path=path.join(savepath, "saves", long_id, long_id) if long_id is not None else None,
                local_rank=local_rank,
                world_size=world_size,
                wandb=wandb,
            ).train()
        else:
            model = ColorMNetTrainer(config, local_rank=local_rank, world_size=world_size, wandb=wandb).train()

        # Load pretrained model if needed
        if base_config.get("load_checkpoint") is not None:
            total_iter = model.load_checkpoint(base_config["load_checkpoint"])
            base_config["load_checkpoint"] = None
            print("Previously trained model loaded!")
        else:
            total_iter = 0

        if network_in_memory is not None:
            print("I am loading network from the previous stage")
            model.load_network_in_memory(network_in_memory)
            network_in_memory = None
        elif base_config.get("load_network") is not None:
            print("I am loading network from a disk, as listed in configuration")
            model.load_network(base_config["load_network"])
            base_config["load_network"] = None

        def worker_init_fn(worker_id):
            worker_seed = torch.initial_seed() % (2**31) + worker_id + local_rank * 100
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        def construct_loader(dataset):
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, rank=local_rank, shuffle=True)
            train_loader = DataLoader(
                dataset,
                config["batch_size"],
                sampler=train_sampler,
                num_workers=config["num_workers"],
                worker_init_fn=worker_init_fn,
                drop_last=True,
            )
            return train_sampler, train_loader

        def renew_DAVIS_Videvo_batch_loader(max_skip, finetune=False):
            davis_dataset = DAVISVidevoDataset(
                train_root,
                train_root,
                max_skip,
                is_bl=False,
                subset=train_videos or None,
                num_frames=config["num_frames"],
                finetune=finetune,
            )
            train_dataset = ConcatDataset([davis_dataset])
            print(f"DAVIS + Videvo dataset size: {len(davis_dataset)}")
            print(f"Concat dataset size: {len(train_dataset)}")
            print(f"Renewed with max_skip={max_skip}")
            return construct_loader(train_dataset)

        max_skip_values = [10, 15, 5, 5]
        increase_skip_fraction = [0.1, 0.3, 0.9, 100]

        train_sampler, train_loader = renew_DAVIS_Videvo_batch_loader(5)
        renew_loader = renew_DAVIS_Videvo_batch_loader

        val_dataset = DAVISTestDataset_221128_TransColorization_batch(
            val_root,
            imset=val_root,
            subset=val_videos or None,
        )

        total_epoch = math.ceil(config["iterations"] / len(train_loader))
        current_epoch = total_iter // len(train_loader)
        print(f"We approximately use {total_epoch} epochs.")
        if stage != "0":
            change_skip_iter = [round(config["iterations"] * f) for f in increase_skip_fraction]
            print(f"The skip value will change approximately at the following iterations: {change_skip_iter[:-1]}")

        finetuning = False
        np.random.seed(np.random.randint(2**30 - 1) + local_rank * 100)
        try:
            while total_iter < config["iterations"] + config["finetune"]:
                train_sampler.set_epoch(current_epoch)
                current_epoch += 1
                print(f"Current epoch: {current_epoch}")

                model.train()
                for data in train_loader:
                    if stage != "0" and total_iter >= change_skip_iter[0]:
                        while total_iter >= change_skip_iter[0]:
                            cur_skip = max_skip_values[0]
                            max_skip_values = max_skip_values[1:]
                            change_skip_iter = change_skip_iter[1:]
                        print(f"Changing skip to cur_skip={cur_skip}")
                        train_sampler, train_loader = renew_loader(cur_skip)
                        break

                    if config["finetune"] > 0 and not finetuning and total_iter >= config["iterations"]:
                        train_sampler, train_loader = renew_loader(cur_skip, finetune=True)
                        finetuning = True
                        model.save_network_interval = 1000
                        break

                    model.do_pass(data, total_iter, val_dataset=val_dataset)
                    total_iter += 1

                    if total_iter >= config["iterations"] + config["finetune"]:
                        break
        finally:
            if not config["debug"] and model.logger is not None and total_iter > 5000:
                model.save_network(total_iter)
                model.save_checkpoint(total_iter)

        network_in_memory = model.model.module.state_dict()

        # Log model artifact if possible
        if local_rank == 0 and config.get("exp_id") and config["exp_id"].lower() != "null":
            save_root = path.join(savepath, "saves", config["exp_id"], config["exp_id"])
            best_path = f"{save_root}_best.pth"
            if os.path.exists(best_path):
                model_artifact = wandb.Artifact(
                    name=f"{config['exp_id']}-model",
                    type="model",
                    description="ColorMNet trained weights",
                    metadata={"exp_id": config["exp_id"], "stage": stage},
                )
                model_artifact.add_file(best_path)
                wb_run.log_artifact(model_artifact)

        # S3 upload of checkpoints
        if local_rank == 0:
            client = get_s3_client(s3_cfg)
            stage_prefix = build_stage_prefix(s3_cfg, f"trainer-s{stage}", wb_run)
            bucket = s3_cfg["bucket"]
            ckpt_dir = path.join(savepath, "saves", config["exp_id"])
            checkpoints = _collect_checkpoints(ckpt_dir)
            s3_info = {
                "stage": f"trainer-s{stage}",
                "prefix": stage_prefix,
                "checkpoint_dir": ckpt_dir,
                "checkpoints": [],
            }
            for ckpt in checkpoints:
                key = f"{stage_prefix}/checkpoints/{ckpt.name}"
                upload_file(client, bucket, str(ckpt), key, s3_cfg)
                s3_info["checkpoints"].append({
                    "file": str(ckpt.name),
                    "uri": f"s3://{bucket}/{key}",
                })
            s3_info_path = write_json_temp(s3_info)
            wb_run.save(s3_info_path, base_path=os.path.dirname(s3_info_path))
            os.unlink(s3_info_path)
            wb_run.summary[f"s3/trainer_s{stage}"] = s3_info

        wb_run.finish()

    distributed.destroy_process_group()


if __name__ == "__main__":
    main()
