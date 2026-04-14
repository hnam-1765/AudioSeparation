"""
Libri2Mix Dataset for audio source separation (8kHz min variant).

Dataset structure inside the zip:
  Libri2Mix/wav8k/min/
    metadata/
      mixture_train-100_mix_clean.csv
      mixture_dev_mix_clean.csv
      mixture_test_mix_clean.csv
      mixture_test_mix_both.csv    # noisy variant
      metrics_*.csv
    metadata_noisy/
      mixture_test_mix_both.csv
      metrics_test_mix_both.csv
    train-100/   ← ~13,900 mixtures
      mix_clean/, s1/, s2/
    dev/         ← ~3,000 mixtures
      mix_clean/, s1/, s2/
    test/        ← ~3,000 mixtures
      mix_clean/, s1/, s2/
    test_noisy/  ← ~3,000 noisy mixtures
      mix_both/, s1/, s2/, noise/

The CSVs use relative paths like:
  ../../Libri2Mix/wav8k/min/train-100/mix_clean/xxx.wav
The parent of "metadata/" is the Libri2Mix/ root.
"""
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from utils.audio import load_wav


# Map between user-facing split names and the actual CSV metadata names
LIBRI2MIX_SPLIT_MAP = {
    "train": "train-100",
    "dev":   "dev",
    "test":  "test",
}

LIBRI2MIX_METADATA_SPLIT_MAP = {
    "train": "train-100",
    "dev":   "dev",
    "test":  "test",
}


def _resolve_librimix2_path(rel_path: str, data_root: Path) -> Path:
    """
    Resolve a path stored in Libri2Mix CSVs to an absolute path.

    CSV paths look like: ../../Libri2Mix/wav8k/min/train-100/mix_clean/xxx.wav
    The parent of Libri2Mix/ is the zip root; data_root is the extracted Libri2Mix/ folder.
    So ../../Libri2Mix/wav8k/...  →  data_root/wav8k/min/...
    """
    rel_path = rel_path.strip()
    # Strip the ../../Libri2Mix/ prefix that CSVs use
    prefix = "../../Libri2Mix/wav8k/min/"
    if rel_path.startswith(prefix):
        rel_path = rel_path[len(prefix):]
    return data_root / rel_path


class Libri2MixDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        partition: str = "train",
        mix_type: str = "mix_clean",
        max_len: int = 80000,
        fs: int = 8000,
        preload: bool = False,
        speed_list: list[float] | None = None,
    ):
        """
        Args:
            root       : path to the Libri2Mix/ root directory (extracted from zip).
            partition  : "train", "dev", or "test"
            mix_type   : "mix_clean" (2 speakers) or "mix_both" (2 speakers + noise, test only)
            max_len    : max samples per utterance (default 10s @ 8kHz)
            fs         : sample rate (default 8000)
            preload    : if True, load all audio into RAM (faster but memory-intensive)
        """
        self.root = Path(root).expanduser().resolve()
        self.partition = partition
        self.mix_type = mix_type
        self.max_len = max_len
        self.fs = fs
        self.preload = preload
        self.speed_list = speed_list or [0.9, 1.0, 1.1]
        self.has_noise = (mix_type == "mix_both")

        # Map partition name → CSV split name
        csv_split = LIBRI2MIX_METADATA_SPLIT_MAP[partition]
        noisy_suffix = "_noisy" if mix_type == "mix_both" else ""
        metadata_subdir = f"metadata{noisy_suffix}"
        csv_name = f"mixture_{csv_split}_{mix_type}.csv"
        csv_path = self.root / metadata_subdir / csv_name

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Libri2Mix metadata not found: {csv_path}\n"
                f"  Expected structure: {self.root}/metadata[_noisy]/mixture_{{split}}_{{mix_type}}.csv"
            )

        self.df = pd.read_csv(csv_path)
        print(f"  [Libri2Mix] {partition}/{mix_type}: {len(self.df)} samples")

        # Resolve all paths from CSV columns
        self.mix_paths = [_resolve_librimix2_path(r["mixture_path"], self.root) for _, r in self.df.iterrows()]
        self.s1_paths  = [_resolve_librimix2_path(r["source_1_path"],  self.root) for _, r in self.df.iterrows()]
        self.s2_paths  = [_resolve_librimix2_path(r["source_2_path"],  self.root) for _, r in self.df.iterrows()]

        # Preload audio if requested
        if preload:
            print(f"  [Libri2Mix] Pre-loading audio into memory...")
            self.mix_data, self.s1_data, self.s2_data = [], [], []
            for i in range(len(self)):
                mix, _ = load_wav(self.mix_paths[i], fs)
                s1,  _ = load_wav(self.s1_paths[i],  fs)
                s2,  _ = load_wav(self.s2_paths[i],  fs)
                self.mix_data.append(mix.astype(np.float32))
                self.s1_data.append(s1.astype(np.float32))
                self.s2_data.append(s2.astype(np.float32))
            print(f"  [Libri2Mix] Done pre-loading.")

    def __len__(self):
        return len(self.df)

    def _trim_pad(self, audio: np.ndarray, start: int | None = None) -> np.ndarray:
        """Trim to max_len with optional start offset, or pad with zeros."""
        if len(audio) > self.max_len:
            if start is None:
                start = random.randint(0, len(audio) - self.max_len)
            audio = audio[start:start + self.max_len]
        elif len(audio) < self.max_len:
            audio = np.pad(audio, (0, self.max_len - len(audio)), mode="constant")
        # Ensure length is divisible by 4 (SepFormer encoder stride = 4)
        if len(audio) % 4 != 0:
            audio = audio[:len(audio) - (len(audio) % 4)]
        return audio

    def _speed_augment(self, audio: np.ndarray, speed: float) -> np.ndarray:
        """Speed change via linear resampling."""
        new_len = int(len(audio) / speed)
        idx = np.linspace(0, len(audio) - 1, new_len)
        return np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)

    def __getitem__(self, idx: int):
        if self.preload:
            mix = self.mix_data[idx].copy()
            s1  = self.s1_data[idx].copy()
            s2  = self.s2_data[idx].copy()
        else:
            mix, _ = load_wav(self.mix_paths[idx], self.fs)
            s1,  _ = load_wav(self.s1_paths[idx],  self.fs)
            s2,  _ = load_wav(self.s2_paths[idx],  self.fs)

        # Random offset for training; deterministic start for val/test
        if self.partition == "train":
            mix = self._trim_pad(mix)
            s1  = self._trim_pad(s1)
            s2  = self._trim_pad(s2)
        else:
            mix = self._trim_pad(mix, start=0)
            s1  = self._trim_pad(s1,  start=0)
            s2  = self._trim_pad(s2,  start=0)

        return {
            "mix":        mix.astype(np.float32),
            "srcs":       [s1.astype(np.float32), s2.astype(np.float32)],
            "key":        self.mix_paths[idx].stem,
            "num_sample": len(mix),
        }


def collate_fn(batch):
    """Collate variable-length audio, sort by length descending."""
    batch = sorted(batch, key=lambda x: x["num_sample"], reverse=True)

    mix_list  = [torch.from_numpy(b["mix"])  for b in batch]
    srcs_list = [b["srcs"] for b in batch]
    lengths   = torch.tensor([b["num_sample"] for b in batch], dtype=torch.long)
    keys      = [b["key"] for b in batch]

    # Pad mixtures
    mix_padded = torch.nn.utils.rnn.pad_sequence(mix_list, batch_first=True)

    # Pad each speaker source
    K = len(srcs_list[0])
    src_padded = [
        torch.nn.utils.rnn.pad_sequence(
            [torch.from_numpy(srcs_list[i][k]) for i in range(len(batch))],
            batch_first=True,
        )
        for k in range(K)
    ]

    return mix_padded, src_padded, lengths, keys


def get_dataloaders(
    root: str,
    batch_size: int = 4,
    num_workers: int = 4,
    max_len: int = 80000,
    fs: int = 8000,
    train_mix_type: str = "mix_clean",
    val_mix_type: str = "mix_clean",
    preload: bool = False,
):
    """Create train (train-100) + val (dev) dataloaders for Libri2Mix."""
    train_ds = Libri2MixDataset(
        root=root,
        partition="train",
        mix_type=train_mix_type,
        max_len=max_len,
        fs=fs,
        preload=preload,
    )
    val_ds = Libri2MixDataset(
        root=root,
        partition="dev",
        mix_type=val_mix_type,
        max_len=max_len,
        fs=fs,
        preload=preload,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader
