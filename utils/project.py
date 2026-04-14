"""Project-wide path and runtime configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def resolve_path(value: str | os.PathLike[str]) -> Path:
    """Resolve a user-provided path to an absolute Path."""
    return Path(value).expanduser().resolve()


def resolve_data_root(data_root: Optional[str] = None) -> Path:
    """
    Resolve the dataset root using, in order:
    1. Explicit CLI argument
    2. AUDIO_SEPARATION_DATA_ROOT env var
    3. ./data/Libri2Mix (new, large dataset)
    4. ./data/MiniLibriMix (legacy, small dataset)
    """
    candidates = [
        data_root,
        os.getenv("AUDIO_SEPARATION_DATA_ROOT"),
        str(PROJECT_ROOT / "data" / "Libri2Mix"),
        str(PROJECT_ROOT / "data" / "MiniLibriMix"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        resolved = resolve_path(candidate)
        if resolved.exists():
            return resolved
    raise FileNotFoundError(
        "Dataset root could not be resolved. "
        "Pass --data_root or set AUDIO_SEPARATION_DATA_ROOT. "
        "Expected: data/Libri2Mix/ or data/MiniLibriMix/."
    )


@dataclass(frozen=True)
class RunPaths:
    project_root: Path
    data_root: Path
    artifacts_root: Path
    model_name: str
    run_name: str
    run_root: Path
    checkpoints_dir: Path
    outputs_dir: Path
    logs_dir: Path
    tensorboard_dir: Path
    evaluation_dir: Path
    wandb_dir: Path
    manifests_dir: Path

    def create(self) -> "RunPaths":
        for path in (
            self.artifacts_root,
            self.run_root,
            self.checkpoints_dir,
            self.outputs_dir,
            self.logs_dir,
            self.tensorboard_dir,
            self.evaluation_dir,
            self.wandb_dir,
            self.manifests_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self


def build_run_paths(
    model_name: str,
    data_root: Path,
    artifacts_root: Optional[str] = None,
    run_name: Optional[str] = None,
) -> RunPaths:
    artifacts_dir = resolve_path(artifacts_root) if artifacts_root else DEFAULT_ARTIFACTS_DIR
    normalized_run_name = run_name or f"{model_name}-default"
    run_root = artifacts_dir / model_name / normalized_run_name
    return RunPaths(
        project_root=PROJECT_ROOT,
        data_root=data_root,
        artifacts_root=artifacts_dir,
        model_name=model_name,
        run_name=normalized_run_name,
        run_root=run_root,
        checkpoints_dir=run_root / "checkpoints",
        outputs_dir=run_root / "outputs",
        logs_dir=run_root / "logs",
        tensorboard_dir=run_root / "tensorboard",
        evaluation_dir=run_root / "evaluation",
        wandb_dir=run_root / "wandb",
        manifests_dir=artifacts_dir / "manifests" / "minilibrimix",
    )
