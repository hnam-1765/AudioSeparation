"""
MiniLibriMix Dataset for audio source separation.

Expected structure:
  /path/to/MiniLibriMix/
    metadata/
      mixture_train_mix_both.csv   (800 rows)
      mixture_train_mix_clean.csv  (800 rows)
      mixture_val_mix_both.csv    (200 rows)
      mixture_val_mix_clean.csv   (200 rows)
    train/
      mix_both/, mix_clean/, noise/, s1/, s2/
    val/
      mix_both/, mix_clean/, noise/, s1/, s2/

Each CSV has columns: mixture_path, source_1_path, source_2_path, [noise_path], length
Paths are relative to MiniLibriMix root.
"""
import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from utils.audio import load_wav


class MiniLibriMixDataset(Dataset):
    def __init__(self, root, partition="train", mix_type="mix_clean",
                 max_len=80000, fs=8000, preload=False,
                 speed_list=None):
        """
        Args:
            root        : path to MiniLibriMix/ directory (contains metadata/ train/ val/)
            partition  : "train" or "val"
            mix_type   : "mix_clean" (2 speakers) or "mix_both" (2 speakers + noise)
            max_len    : max samples per utterance (default 10s @ 8kHz)
            fs         : sample rate (default 8000)
            preload    : if True, load all audio into RAM (faster but uses ~1GB RAM)
        """
        self.root = root
        self.partition = partition
        self.mix_type = mix_type
        self.max_len = max_len
        self.fs = fs
        self.preload = preload
        self.speed_list = speed_list or [0.9, 1.0, 1.1]
        self.has_noise = (mix_type == "mix_both")

        # Load CSV metadata
        csv_name = f"mixture_{partition}_mix_{mix_type}.csv"
        csv_path = os.path.join(root, "metadata", csv_name)
        self.df = pd.read_csv(csv_path)
        print(f"  [MiniLibriMix] {partition}/{mix_type}: {len(self.df)} samples")

        # Resolve absolute paths
        def abs_path(rel):
            """Convert path like 'train/s1/foo.wav' to full absolute path."""
            rel = rel.strip()
            # The CSV may contain paths like "MiniLibriMix/train/s1/foo.wav"
            # or "train/s1/foo.wav" — strip the MiniLibriMix/ prefix
            if rel.startswith("MiniLibriMix/"):
                rel = rel[len("MiniLibriMix/"):]
            return os.path.join(root, rel)

        self.mix_paths = [abs_path(r["mixture_path"]) for _, r in self.df.iterrows()]
        self.s1_paths  = [abs_path(r["source_1_path"])  for _, r in self.df.iterrows()]
        self.s2_paths  = [abs_path(r["source_2_path"])  for _, r in self.df.iterrows()]

        # Preload audio if requested
        if preload:
            print(f"  [MiniLibriMix] Pre-loading audio into memory...")
            self.mix_data, self.s1_data, self.s2_data = [], [], []
            for i in range(len(self)):
                mix, _ = load_wav(self.mix_paths[i], fs)
                s1,  _ = load_wav(self.s1_paths[i],  fs)
                s2,  _ = load_wav(self.s2_paths[i],  fs)
                self.mix_data.append(mix.astype(np.float32))
                self.s1_data.append(s1.astype(np.float32))
                self.s2_data.append(s2.astype(np.float32))
            print(f"  [MiniLibriMix] Done pre-loading.")

    def __len__(self):
        return len(self.df)

    def _trim_pad(self, audio, start=None):
        """Trim to max_len with optional start offset, or pad with zeros."""
        if len(audio) > self.max_len:
            if start is None:
                start = random.randint(0, len(audio) - self.max_len)
            audio = audio[start:start + self.max_len]
        elif len(audio) < self.max_len:
            audio = np.pad(audio, (0, self.max_len - len(audio)), mode='constant')
        # Ensure divisible by 4 (SepFormer encoder stride = 4 per stage × 4 stages)
        if len(audio) % 4 != 0:
            audio = audio[:len(audio) - (len(audio) % 4)]
        return audio

    def _speed_augment(self, audio, speed):
        """Speed change via linear resampling."""
        new_len = int(len(audio) / speed)
        idx = np.linspace(0, len(audio) - 1, new_len)
        return np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)

    def __getitem__(self, idx):
        if self.preload:
            mix = self.mix_data[idx].copy()
            s1  = self.s1_data[idx].copy()
            s2  = self.s2_data[idx].copy()
        else:
            mix, _ = load_wav(self.mix_paths[idx], self.fs)
            s1,  _ = load_wav(self.s1_paths[idx],  self.fs)
            s2,  _ = load_wav(self.s2_paths[idx],  self.fs)

        # Random start offset for training
        if self.partition == "train":
            mix = self._trim_pad(mix)
            s1  = self._trim_pad(s1)
            s2  = self._trim_pad(s2)
        else:
            # Validation/test: deterministic — take from start
            mix = self._trim_pad(mix, start=0)
            s1  = self._trim_pad(s1,  start=0)
            s2  = self._trim_pad(s2,  start=0)

        return {
            "mix":       mix.astype(np.float32),
            "srcs":      [s1.astype(np.float32), s2.astype(np.float32)],
            "key":       os.path.basename(self.mix_paths[idx]),
            "num_sample": len(mix),
        }


def collate_fn(batch):
    """Collate variable-length audio, sort by length descending."""
    batch = sorted(batch, key=lambda x: x['num_sample'], reverse=True)

    mix_list   = [torch.from_numpy(b['mix'])  for b in batch]
    srcs_list  = [b['srcs']  for b in batch]
    lengths    = torch.tensor([b['num_sample'] for b in batch], dtype=torch.long)
    keys       = [b['key'] for b in batch]

    # Pad mixture
    mix_padded = torch.nn.utils.rnn.pad_sequence(mix_list, batch_first=True)

    # Pad each speaker
    K = len(srcs_list[0])
    src_padded = [
        torch.nn.utils.rnn.pad_sequence(
            [torch.from_numpy(srcs_list[i][k]) for i in range(len(batch))],
            batch_first=True)
        for k in range(K)
    ]

    return mix_padded, src_padded, lengths, keys


def get_dataloaders(root, batch_size=4, num_workers=4,
                    max_len=80000, fs=8000,
                    train_mix_type="mix_clean", val_mix_type="mix_clean",
                    preload=False):
    """Create train + val dataloaders."""
    train_ds = MiniLibriMixDataset(
        root=root, partition="train", mix_type=train_mix_type,
        max_len=max_len, fs=fs, preload=preload)

    val_ds = MiniLibriMixDataset(
        root=root, partition="val", mix_type=val_mix_type,
        max_len=max_len, fs=fs, preload=preload)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=False)

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=False)

    return train_loader, val_loader
