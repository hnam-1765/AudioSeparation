"""
Kaldi-style manifest generation for Libri2Mix (8kHz min variant).

Produces SCP files compatible with the SepFormer pipeline:
  manifest_root/
    train/
      tr_mix.scp   tr_s1.scp   tr_s2.scp
    valid/
      cv_mix.scp   cv_s1.scp   cv_s2.scp
    test/
      tt_mix.scp   tt_s1.scp   tt_s2.scp
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


# CSV paths are relative like ../../Libri2Mix/wav8k/min/train-100/mix_clean/xxx.wav
# Strip this prefix to get the path relative to the Libri2Mix/ root.
_LIBRI2MIX_CSV_PREFIX = "../../Libri2Mix/wav8k/min/"


def _strip_csv_prefix(rel_path: str) -> Path:
    rel_path = rel_path.strip()
    if rel_path.startswith(_LIBRI2MIX_CSV_PREFIX):
        rel_path = rel_path[len(_LIBRI2MIX_CSV_PREFIX):]
    return Path(rel_path)


def create_librimix2_scp(
    data_root: str | Path,
    scp_root: str | Path,
    mix_type: str = "mix_clean",
) -> Path:
    """Generate Kaldi-style SCP manifests for Libri2Mix.

    Args:
        data_root : extracted Libri2Mix/ directory (contains wav8k/min/...)
        scp_root  : output directory for SCP files
        mix_type  : "mix_clean" or "mix_both"

    Returns:
        Path to the scp_root directory
    """
    data_root = Path(data_root).expanduser().resolve()
    scp_root  = Path(scp_root).expanduser().resolve()

    marker = scp_root / "train" / "tr_mix.scp"
    if marker.exists():
        return scp_root

    # Map user-facing split names → (output_subdir, scp_prefix, CSV split name)
    split_map = {
        "train": ("train", "tr", "train-100"),
        "dev":   ("valid", "cv", "dev"),
    }
    test_csv_path = data_root / "metadata_noisy" / f"mixture_test_{mix_type}.csv"
    split_map["test"] = ("test", "tt", "test")

    noisy_suffix = "_noisy" if mix_type == "mix_both" else ""
    metadata_subdir = f"metadata{noisy_suffix}"

    for split_name, (output_subdir, prefix, csv_split) in split_map.items():
        noisy_extra = "/metadata_noisy" if mix_type == "mix_both" else "/metadata"
        csv_path = data_root / f"{noisy_extra}" / f"mixture_{csv_split}_{mix_type}.csv"

        # For clean mix_both fallback, try metadata/ first
        if not csv_path.exists():
            csv_path = data_root / "metadata" / f"mixture_{csv_split}_{mix_type}.csv"

        if not csv_path.exists():
            if split_name == "test":
                continue
            raise FileNotFoundError(f"Libri2Mix metadata not found: {csv_path}")

        df = pd.read_csv(csv_path)
        target_dir = scp_root / output_subdir
        target_dir.mkdir(parents=True, exist_ok=True)

        output_specs = {
            "mixture_path":  f"{prefix}_mix.scp",
            "source_1_path":  f"{prefix}_s1.scp",
            "source_2_path":  f"{prefix}_s2.scp",
        }

        for column_name, filename in output_specs.items():
            output_path = target_dir / filename
            with open(output_path, "w", encoding="utf-8") as f:
                for _, row in df.iterrows():
                    relative = _strip_csv_prefix(row[column_name])
                    absolute = data_root / relative
                    key = absolute.stem
                    f.write(f"{key}\t{absolute}\n")

    return scp_root
