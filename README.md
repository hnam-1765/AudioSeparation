# Audio Separation

Production-oriented refactor of a small research codebase for single-channel speech separation on **MiniLibriMix**, with two bundled model families:

- `SepFormer`-style encoder-separator-decoder pipeline
- `MossFormer2`-style dual-path transformer pipeline

The repository is still lightweight and research-friendly, but the runtime layout is now cleaner for local experiments, sharing, and publishing on GitHub:

- no hardcoded absolute dataset or output paths
- centralized runtime path handling
- generated SCP manifests for SepFormer under a managed artifact directory
- clearer training/inference entry points
- fewer fragile `sys.path` hacks
- improved README and dependency story

## What This Repo Does

This repository trains and evaluates speech separation models on pre-mixed two-speaker MiniLibriMix audio. It provides:

- `train.py` as the main entry point for both SepFormer and MossFormer2
- `inference.py` for checkpoint-based waveform separation
- dataset loaders for MiniLibriMix CSV metadata and SepFormer SCP manifests
- basic experiment logging, checkpointing, TensorBoard, and optional Weights & Biases integration

## Repository Layout

```text
audio_separation/
├── train.py
├── inference.py
├── engine.py
├── requirements.txt
├── .gitignore
├── configs/
│   └── sepformer.yaml
├── datasets/
│   ├── __init__.py
│   └── minilibrimix.py
├── models/
│   ├── mossformer2/
│   │   ├── __init__.py
│   │   ├── conv_module.py
│   │   ├── fsmn.py
│   │   ├── model.py
│   │   ├── mossformer2.py
│   │   ├── normalization.py
│   │   └── Transformer.py
│   └── sepformer/
│       ├── configs.yaml
│       ├── dataset.py
│       ├── engine.py
│       ├── main.py
│       ├── model.py
│       └── modules/
│           ├── __init__.py
│           ├── module.py
│           └── network.py
├── sepformer_utils/
│   ├── __init__.py
│   ├── decorators.py
│   ├── functions.py
│   ├── util_dataset.py
│   ├── util_engine.py
│   ├── util_implement.py
│   ├── util_system.py
│   └── implements/
│       ├── criterions.py
│       ├── mir_eval_stub.py
│       ├── optimizers.py
│       └── schedulers.py
└── utils/
    ├── __init__.py
    ├── audio.py
    ├── config.py
    ├── losses.py
    ├── manifests.py
    ├── metrics.py
    ├── project.py
    └── train_utils.py
```

## Dataset Layout

The code expects a MiniLibriMix directory with CSV metadata and audio files similar to:

```text
MiniLibriMix/
├── metadata/
│   ├── mixture_train_mix_clean.csv
│   ├── mixture_val_mix_clean.csv
│   └── mixture_test_mix_clean.csv
├── train/
│   ├── mix_clean/
│   ├── s1/
│   └── s2/
├── val/
│   ├── mix_clean/
│   ├── s1/
│   └── s2/
└── test/
    ├── mix_clean/
    ├── s1/
    └── s2/
```

Each CSV should contain at least:

- `mixture_path`
- `source_1_path`
- `source_2_path`

Paths may be relative to the dataset root, with or without a leading `MiniLibriMix/` prefix.

## Installation

Create an environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you do not need all optional metrics or logging integrations, you can install only the core packages:

```bash
pip install numpy pandas pyyaml scipy torch tqdm loguru tensorboard
```

## Configuration and Path Handling

The repository resolves the dataset root in this order:

1. `--data_root`
2. `AUDIO_SEPARATION_DATA_ROOT`
3. `./data/MiniLibriMix` relative to the repository root

Runtime artifacts are written under:

```text
artifacts/<model_name>/<run_name>/
```

This includes:

- checkpoints
- TensorBoard logs
- evaluation CSVs
- separated waveform outputs
- Weights & Biases metadata

SepFormer SCP manifests are generated automatically under:

```text
artifacts/manifests/minilibrimix/
```

## Training

### MossFormer2

```bash
python train.py \
  --model mossformer2 \
  --data_root /path/to/MiniLibriMix \
  --epochs 50 \
  --batch_size 4 \
  --gpu 0 \
  --run_name mossformer2-baseline
```

### SepFormer

```bash
python train.py \
  --model sepformer \
  --data_root /path/to/MiniLibriMix \
  --epochs 50 \
  --batch_size 2 \
  --gpu 0 \
  --run_name sepformer-baseline
```

### Optional Weights & Biases

```bash
python train.py \
  --model mossformer2 \
  --data_root /path/to/MiniLibriMix \
  --wandb_project audio-separation \
  --wandb_name mossformer2-exp1
```

Disable W&B explicitly:

```bash
python train.py --model mossformer2 --no_wandb
```

## Inference

```bash
python inference.py \
  --model mossformer2 \
  --input /path/to/mix.wav \
  --checkpoint artifacts/mossformer2/mossformer2-baseline/checkpoints/best.pt \
  --output_dir ./outputs
```

Or with SepFormer:

```bash
python inference.py \
  --model sepformer \
  --input /path/to/mix.wav \
  --checkpoint artifacts/sepformer/sepformer-baseline/checkpoints/epoch.0050.pth \
  --output_dir ./outputs
```

## Key CLI Arguments

### Shared

- `--model`: `sepformer` or `mossformer2`
- `--data_root`: dataset root
- `--artifacts_root`: override artifact directory
- `--run_name`: subdirectory name for outputs/checkpoints/logs
- `--epochs`: number of training epochs
- `--batch_size`: batch size
- `--num_workers`: data loader workers
- `--gpu`: GPU id string such as `0`
- `--no_cuda`: force CPU
- `--no_wandb`: disable W&B

### MossFormer2-specific runtime options

- `--mix_type`: `mix_clean` or `mix_both`
- `--sample_rate`: audio sampling rate
- `--max_len`: max waveform length in samples
- `--preload_audio`: preload waveforms into memory

### SepFormer-specific runtime options

- `--engine_mode`: `train` or a mode containing `test`
- `--out_wav_dir`: override directory for test-time waveform dumps

## Outputs

By default, a run such as:

```bash
python train.py --model mossformer2 --run_name demo
```

produces artifacts like:

```text
artifacts/
├── manifests/minilibrimix/
└── mossformer2/demo/
    ├── checkpoints/
    ├── evaluation/
    ├── logs/
    ├── outputs/
    ├── tensorboard/
    └── wandb/
```

## Notes on the Bundled SepFormer Code

The SepFormer branch in this repository is adapted from an older research-style code layout. Some legacy files are still present, including `models/sepformer/configs.yaml`, but the main training flow now reads runtime configuration from:

```text
configs/sepformer.yaml
```

That keeps the public entry point cleaner while minimizing disruption to the original model implementation.

## Known Limitations

- This is still a research codebase, not a packaged PyPI project.
- There are committed `__pycache__` files in the repository history that should be removed before a clean public release.
- Metric coverage depends on optional packages such as `pesq` and `pystoi`.
- SepFormer internals are still more complex and less uniform than the MossFormer2 path.

## Suggested Next Cleanup Steps

- remove tracked `__pycache__` and `.pyc` files from version control
- consider moving `models/sepformer/configs.yaml` out of the tree or marking it deprecated
- pin exact dependency versions once you settle on a training environment
- add smoke tests for import, one-batch training, and inference
- add sample experiment configs for reproducibility
