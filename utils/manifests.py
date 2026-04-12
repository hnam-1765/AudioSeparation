"""Manifest helpers for datasets used in the project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def normalize_dataset_relative_path(path_value: str) -> str:
    normalized = path_value.strip()
    if normalized.startswith("MiniLibriMix/"):
        normalized = normalized[len("MiniLibriMix/") :]
    return normalized


def create_minilibrimix_scp(data_root: str | Path, scp_root: str | Path, mix_type: str = "mix_clean") -> Path:
    """Generate Kaldi-style SCP manifests for the SepFormer pipeline."""
    data_root = Path(data_root).expanduser().resolve()
    scp_root = Path(scp_root).expanduser().resolve()
    marker = scp_root / "train" / "tr_mix.scp"
    if marker.exists():
        return scp_root

    split_to_prefix = {
        "train": ("train", "tr"),
        "val": ("valid", "cv"),
    }
    test_metadata = data_root / "metadata" / f"mixture_test_{mix_type}.csv"
    split_to_prefix["test"] = ("test", "tt") if test_metadata.exists() else ("valid", "tt")

    for split_name, (output_subdir, prefix) in split_to_prefix.items():
        csv_path = data_root / "metadata" / f"mixture_{split_name}_{mix_type}.csv"
        if not csv_path.exists():
            if split_name == "test":
                continue
            raise FileNotFoundError(f"MiniLibriMix metadata not found: {csv_path}")

        df = pd.read_csv(csv_path)
        target_dir = scp_root / output_subdir
        target_dir.mkdir(parents=True, exist_ok=True)

        output_specs = {
            "mixture_path": f"{prefix}_mix.scp",
            "source_1_path": f"{prefix}_s1.scp",
            "source_2_path": f"{prefix}_s2.scp",
        }
        for column_name, filename in output_specs.items():
            output_path = target_dir / filename
            with open(output_path, "w", encoding="utf-8") as handle:
                for _, row in df.iterrows():
                    relative_path = normalize_dataset_relative_path(row[column_name])
                    absolute_path = data_root / relative_path
                    key = absolute_path.stem
                    handle.write(f"{key}\t{absolute_path}\n")

    return scp_root
