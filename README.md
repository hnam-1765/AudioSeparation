# 🎧 Audio Source Separation

Train and evaluate **SepFormer** and **MossFormer2** for single-channel two-speaker speech separation on **MiniLibriMix** — offline-friendly, no internet required.

---

## 🗂️ Repository Layout

```
audio_separation/
├── train.py                  # Unified entry point (SepFormer + MossFormer2)
├── inference.py              # Checkpoint-based separation
├── engine.py                 # Shared inference engine
├── requirements.txt
├── LICENSE
├── configs/
│   ├── sepformer.yaml        # SepFormer Large config
│   └── sepformer_base.yaml   # SepFormer Base config  ← recommended for MiniLibriMix
├── datasets/
│   ├── __init__.py
│   └── minilibrimix.py       # MiniLibriMix loader
├── models/
│   ├── __init__.py
│   ├── mossformer2/          # MossFormer2 architecture
│   └── sepformer/            # SepFormer architecture (encoder-separator-decoder)
│       ├── configs.yaml      # ⚠️  Deprecated; use configs/ instead
│       ├── dataset.py        # SCP manifest loader
│       ├── engine.py         # Training engine
│       ├── main.py
│       ├── model.py
│       └── modules/
│           ├── module.py     # Core building blocks
│           └── network.py    # GCFN, CLA, EGA, attention layers
├── sepformer_utils/          # SepFormer-specific training utilities
│   ├── decorators.py
│   ├── functions.py
│   ├── util_dataset.py
│   ├── util_engine.py
│   ├── util_implement.py
│   ├── util_system.py
│   └── implements/
│       ├── criterions.py     # PIT loss variants
│       ├── optimizers.py
│       └── schedulers.py
└── utils/
    ├── __init__.py
    ├── audio.py              # WAV load/save helpers
    ├── config.py             # YAML config helpers
    ├── losses.py
    ├── manifests.py          # SCP manifest generation for MiniLibriMix
    ├── metrics.py            # SI-SDR, SDR, PESQ, STOI
    ├── project.py
    └── train_utils.py
```

---

## ⚙️ Configurations

### SepFormer Base (`configs/sepformer_base.yaml`) — Recommended for MiniLibriMix

Lightweight variant tuned for the small MiniLibriMix dataset (800 train / 200 val samples).

| Parameter | Value |
|---|---|
| `dynamic_mixing` | `false` (uses pre-mixed files) |
| Feature dim (F) | **128** |
| Dropout | **0.05** |
| Learning rate | **1e-3** |
| Batch size | **16** |
| Max epochs | **50** |
| Test at epoch | 50 only |

### SepFormer Large (`configs/sepformer.yaml`)

Full-size variant from the original SepFormer paper. Designed for WSJ-style datasets with dynamic on-the-fly mixing (`dynamic_mixing: true`).

| Parameter | Value |
|---|---|
| `dynamic_mixing` | `true` |
| Feature dim (F) | 256 |
| Dropout | 0.1 |
| Learning rate | 2e-4 |
| Batch size | 2 |
| Max epochs | 200 |

---

## 📦 Dataset

### MiniLibriMix

Download from [HuggingFace datasets](https://huggingface.co/datasets/minilibrimix) or [GitHub](https://github.com/JorisCosentino/LibriMix).

```
MiniLibriMix/
├── metadata/
│   ├── mixture_train_mix_clean.csv
│   ├── mixture_val_mix_clean.csv
│   └── mixture_test_mix_clean.csv
├── train/  { mix_clean/, mix_both/, noise/, s1/, s2/ }
└── val/    { mix_clean/, mix_both/, noise/, s1/, s2/ }
```

Each CSV has columns: `mixture_path`, `source_1_path`, `source_2_path`, `noise_path`, `length`.
SCP manifests are generated automatically — no manual setup needed.

---

## 🚀 Quick Start

### 1 — Install dependencies

```bash
# CPU only
pip install numpy pandas scipy torch tqdm pyyaml

# With full metrics (PESQ, STOI, Weights & Biases, TensorBoard)
pip install -r requirements.txt
```

### 2 — Train SepFormer Base (recommended)

```bash
python train.py \
  --model sepformer \
  --sepformer_config base \
  --data_root /path/to/MiniLibriMix \
  --epochs 50 \
  --batch_size 16 \
  --gpu 0 \
  --no_wandb
```

### 3 — Train MossFormer2

```bash
python train.py \
  --model mossformer2 \
  --data_root /path/to/MiniLibriMix \
  --epochs 50 \
  --batch_size 4 \
  --gpu 0 \
  --no_wandb
```

### 4 — Inference

```bash
python inference.py \
  --model sepformer \
  --input /path/to/mix.wav \
  --checkpoint artifacts/sepformer/sepformer-base-bs16/checkpoints/best.pt \
  --output_dir ./outputs
```

---

## 📁 Outputs

All artifacts are written under `artifacts/<model_name>/<run_name>/`:

```
artifacts/
├── manifests/minilibrimix/   # auto-generated SCP files
└── sepformer/
    └── sepformer-base-bs16/
        ├── checkpoints/      # best.pt, latest.pt, epoch.*.pth
        ├── evaluation/        # test_metrics.csv
        ├── logs/              # system_log.log
        ├── outputs/           # separated wav files
        └── tensorboard/
```

---

## 🔧 Key CLI Arguments

### Shared

| Argument | Description | Default |
|---|---|---|
| `--model` | `sepformer` or `mossformer2` | `mossformer2` |
| `--data_root` | Path to MiniLibriMix | — |
| `--epochs` | Number of training epochs | 50 |
| `--batch_size` | Batch size | 4 |
| `--gpu` | GPU id (`0`, `1`, `0,1`) | `0` |
| `--no_cuda` | Force CPU | — |
| `--no_wandb` | Disable Weights & Biases | off |
| `--wandb_project` | W&B project name | `audio-separation` |
| `--run_name` | Subdirectory name | auto |

### SepFormer only

| Argument | Description | Default |
|---|---|---|
| `--sepformer_config` | `base` (recommended) or `large` | `base` |
| `--mix_type` | `mix_clean` or `mix_both` | `mix_clean` |
| `--engine_mode` | `train` or mode containing `test` | `train` |
| `--out_wav_dir` | Override test wav output dir | auto |

### MossFormer2 only

| Argument | Description | Default |
|---|---|---|
| `--mix_type` | `clean` or `both` | `clean` |
| `--sample_rate` | Audio sample rate | 8000 |
| `--max_len` | Max waveform samples | 80000 |
| `--preload_audio` | Preload all audio to RAM | off |
| `--lr` | Learning rate | 1e-3 |

---

## 📊 Metrics

Evaluation computes per-sample and aggregate:

| Metric | Description |
|---|---|
| **SI-SDR** | Scale-Invariant Signal-to-Distortion Ratio |
| **SDR** | Signal-to-Distortion Ratio |
| **SI-SNR** | Scale-Invariant SNR |
| **SNR** | Signal-to-Noise Ratio |
| **PESQ** | Perceptual Evaluation of Speech Quality |
| **STOI** | Short-Time Objective Intelligibility |

---

## ⚠️ Notes

- `models/sepformer/configs.yaml` is **deprecated** — use `configs/sepformer.yaml` or `configs/sepformer_base.yaml`.
- The codebase is offline-friendly: no pip installs from internet required during training on Kaggle or similar air-gapped environments.
- The SepFormer code in `models/sepformer/` was adapted from a research-style layout; `train.py` is the clean public entry point.
