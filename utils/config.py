"""Lightweight configuration helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from utils.project import RunPaths


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def apply_sepformer_runtime_overrides(
    raw_config: dict[str, Any],
    paths: RunPaths,
    batch_size: int,
    num_workers: int,
    max_epoch: int,
    data_parallel_gpu_ids: str,
    sepformer_config: str = "base",
) -> dict[str, Any]:
    """Inject runtime-dependent paths into the SepFormer YAML config.

    Args:
        sepformer_config: "base" (smaller, for MiniLibriMix) or "large".
            Only max_epoch from the YAML is used when sepformer_config="large";
            for "base", max_epoch from the YAML (50) is respected.
    """
    config = deepcopy(raw_config["config"])
    config["dataset"]["scp_dir"] = str(paths.manifests_dir)
    config["dataloader"]["batch_size"] = batch_size
    config["dataloader"]["num_workers"] = num_workers
    # For "large" config, always honour the CLI --epochs override;
    # for "base", the YAML already has the right default (50) so we skip
    # overwriting max_epoch so users can still pass --epochs if they want.
    if sepformer_config == "large":
        config["engine"]["max_epoch"] = max_epoch
    config["engine"]["gpuid"] = data_parallel_gpu_ids
    config.setdefault("runtime", {})
    config["runtime"].update(
        {
            "checkpoints_dir": str(paths.checkpoints_dir),
            "logs_dir": str(paths.logs_dir),
            "tensorboard_dir": str(paths.tensorboard_dir),
            "evaluation_dir": str(paths.evaluation_dir),
            "outputs_dir": str(paths.outputs_dir),
            "wandb_dir": str(paths.wandb_dir),
        }
    )
    return config
