"""Unified training entry point for SepFormer and MossFormer2."""

from __future__ import annotations

import argparse
from itertools import permutations
from types import SimpleNamespace

import torch
from tqdm import tqdm

from datasets.minilibrimix import get_dataloaders
from models.mossformer2 import MossFormer2
from utils.config import apply_sepformer_runtime_overrides, load_yaml_config
from utils.manifests import create_minilibrimix_scp
from utils.metrics import pesq, stoi
from utils.project import PROJECT_ROOT, build_run_paths, resolve_data_root
from utils.train_utils import save_checkpoint


def train_sepformer(args):
    """Train the bundled SepFormer pipeline with runtime path overrides."""
    from models.sepformer import main as sepformer_main

    data_root = resolve_data_root(args.data_root)

    sepformer_cfg_name = getattr(args, "sepformer_config", "base")
    sepformer_yaml_map = {
        "base": "sepformer_base.yaml",
        "large": "sepformer.yaml",
    }
    sepformer_yaml = sepformer_yaml_map.get(sepformer_cfg_name, "sepformer_base.yaml")
    run_name_suffix = f"-{sepformer_cfg_name}" if sepformer_cfg_name != "base" else ""

    paths = build_run_paths(
        model_name="sepformer",
        data_root=data_root,
        artifacts_root=args.artifacts_root,
        run_name=args.run_name or f"sepformer{run_name_suffix}-bs{args.batch_size}",
    ).create()
    create_minilibrimix_scp(data_root=data_root, scp_root=paths.manifests_dir, mix_type=args.mix_type)

    config = apply_sepformer_runtime_overrides(
        raw_config=load_yaml_config(PROJECT_ROOT / "configs" / sepformer_yaml),
        paths=paths,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epoch=args.epochs,
        data_parallel_gpu_ids=args.gpu,
        sepformer_config=sepformer_cfg_name,
    )

    sepformer_args = SimpleNamespace(
        engine_mode=args.engine_mode,
        out_wav_dir=args.out_wav_dir or str(paths.outputs_dir),
        sample_file=None,
        no_wandb=args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        config=config,
    )
    sepformer_main.main(sepformer_args)


def train_mossformer2(args):
    """Train MossFormer2 on MiniLibriMix."""
    data_root = resolve_data_root(args.data_root)
    paths = build_run_paths(
        model_name="mossformer2",
        data_root=data_root,
        artifacts_root=args.artifacts_root,
        run_name=args.run_name or f"mossformer2-bs{args.batch_size}",
    ).create()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"[MossFormer2] Device: {device}")
    print(f"[MossFormer2] Data root: {data_root}")
    print(f"[MossFormer2] Artifacts: {paths.run_root}")

    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project or "audio-separation",
                name=args.wandb_name or paths.run_name,
                config={
                    "model": "MossFormer2",
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "epochs": args.epochs,
                    "max_len": args.max_len,
                    "num_spks": args.num_spks,
                    "optimizer": "AdamW",
                    "data_root": str(data_root),
                },
                dir=str(paths.wandb_dir),
            )
            print(f"[WandB] Initialized: {wandb_run.url}")
        except Exception as exc:
            print(f"[WandB] init failed: {exc} — continuing without WandB")

    model = MossFormer2(
        num_spks=args.num_spks,
        encoder_kernel_size=16,
        encoder_out_nchannels=512,
        encoder_in_nchannels=1,
        masknet_chunksize=250,
        masknet_numlayers=1,
        masknet_norm="ln",
        intra_numlayers=24,
        intra_nhead=8,
        intra_dffn=1024,
    ).to(device)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    train_loader, val_loader = get_dataloaders(
        root=str(data_root),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_len=args.max_len,
        fs=args.sample_rate,
        train_mix_type=args.mix_type,
        val_mix_type=args.mix_type,
        preload=args.preload_audio,
    )
    print(f"  Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.8,
        patience=3,
        min_lr=1e-7,
    )

    def _sisdr(est, ref, eps=1e-8):
        r_zm = ref - ref.mean(-1, keepdim=True)
        e_zm = est - est.mean(-1, keepdim=True)
        scale = (e_zm * r_zm).sum(-1) / (r_zm.pow(2).sum(-1) + eps)
        r_s = torch.clamp(scale, min=1e-2) * r_zm
        num = r_s.pow(2).sum(-1)
        den = (e_zm - r_s).pow(2).sum(-1) + eps
        return 10 * torch.log10(eps + num / den)

    def _sdr(est, ref, eps=1e-8):
        num = ref.pow(2).sum(-1)
        den = (est - ref).pow(2).sum(-1) + eps
        return 10 * torch.log10(eps + num / den)

    def sisnr(est, tgt, eps=1e-8):
        tgt_m = tgt - tgt.mean(-1, keepdim=True)
        est_m = est - est.mean(-1, keepdim=True)
        scale = (est_m * tgt_m).sum(-1, keepdim=True) / (tgt_m.pow(2).sum(-1, keepdim=True) + eps)
        tgt_s = torch.clamp(scale, min=1e-2) * tgt_m
        num = tgt_s.norm(2, -1)
        den = (est_m - tgt_s).norm(2, -1)
        return -20 * torch.log10(eps + num / (den + eps))

    def pit_loss(ests, tgts):
        scores = []
        for perm in permutations(range(len(ests))):
            values = [sisnr(ests[s], tgts[t]) for s, t in enumerate(perm)]
            scores.append(sum(values).mean() / len(ests))
        return min(scores)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"[Epoch {epoch}] Train", dynamic_ncols=True)
        for mix, srcs, lens, _ in train_bar:
            mix = mix.to(device)
            srcs_t = [src.to(device) for src in srcs]

            optimizer.zero_grad()
            out = model(mix)
            ests = [out[:, :, speaker_idx] for speaker_idx in range(args.num_spks)]
            loss = pit_loss(ests, srcs_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix(sisnr=f"{loss.item():.2f}")

            if wandb_run is not None:
                wandb_run.log({"train/loss_step": loss.item()})

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        tot_si_sdr = 0.0
        tot_sdr = 0.0
        tot_pesq = 0.0
        tot_stoi = 0.0
        n_samples = 0

        with torch.inference_mode():
            valid_bar = tqdm(val_loader, desc=f"[Epoch {epoch}] Valid", dynamic_ncols=True)
            for mix, srcs, lens, _ in valid_bar:
                mix = mix.to(device)
                srcs_t = [src.to(device) for src in srcs]
                out = model(mix)
                ests = [out[:, :, speaker_idx] for speaker_idx in range(args.num_spks)]
                loss = pit_loss(ests, srcs_t)
                val_loss += loss.item()

                for batch_index in range(mix.size(0)):
                    sample_length = int(lens[batch_index])
                    for speaker_idx in range(args.num_spks):
                        estimate = ests[speaker_idx][batch_index, :sample_length].float().cpu()
                        reference = srcs_t[speaker_idx][batch_index, :sample_length].float().cpu()
                        tot_si_sdr += _sisdr(estimate, reference).item()
                        tot_sdr += _sdr(estimate, reference).item()
                        estimate_np = estimate.numpy()
                        reference_np = reference.numpy()
                        tot_pesq += pesq(reference_np, estimate_np)
                        tot_stoi += stoi(reference_np, estimate_np)
                        n_samples += 1

                valid_bar.set_postfix(
                    sisnr=f"{loss.item():.2f}",
                    si_sdr=f"{tot_si_sdr / max(n_samples, 1):.2f}",
                    sdr=f"{tot_sdr / max(n_samples, 1):.2f}",
                )

        val_loss /= len(val_loader)
        avg_si_sdr = tot_si_sdr / max(n_samples, 1)
        avg_sdr = tot_sdr / max(n_samples, 1)
        avg_pesq = tot_pesq / max(n_samples, 1)
        avg_stoi = tot_stoi / max(n_samples, 1)

        scheduler.step(val_loss)
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d} | "
            f"TrainSISNR={train_loss:7.2f}dB | ValSISNR={val_loss:7.2f}dB | "
            f"SI-SDR={avg_si_sdr:7.2f}dB | SDR={avg_sdr:7.2f}dB | "
            f"PESQ={avg_pesq:.3f} | STOI={avg_stoi:.3f} | "
            f"LR={lr:.2e} {'*BEST' if is_best else ''}"
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "valid/loss": val_loss,
                    "valid/si_sdr": avg_si_sdr,
                    "valid/sdr": avg_sdr,
                    "valid/pesq": avg_pesq,
                    "valid/stoi": avg_stoi,
                    "train/sisnr_db": -train_loss,
                    "valid/sisnr_db": -val_loss,
                    "learning_rate": lr,
                    "is_best": int(is_best),
                }
            )

        latest_ckpt_path = paths.checkpoints_dir / "latest.pt"
        save_checkpoint(epoch, model, optimizer, str(latest_ckpt_path), best_loss=best_val)
        if is_best:
            best_ckpt_path = paths.checkpoints_dir / "best.pt"
            save_checkpoint(epoch, model, optimizer, str(best_ckpt_path), best_loss=best_val)

    if wandb_run is not None:
        wandb_run.finish()

    print(f"\nDone! Best val SISNR: {-best_val:.2f}dB")


def main():
    parser = argparse.ArgumentParser(description="Train SepFormer or MossFormer2 on MiniLibriMix")
    parser.add_argument("--model", type=str, default="mossformer2", choices=["sepformer", "mossformer2"])
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to MiniLibriMix. If omitted, use AUDIO_SEPARATION_DATA_ROOT or ./data/MiniLibriMix.",
    )
    parser.add_argument("--artifacts_root", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--mix_type", type=str, default="mix_clean", choices=["mix_clean", "mix_both"])
    parser.add_argument("--sample_rate", type=int, default=8000)
    parser.add_argument("--max_len", type=int, default=80000)
    parser.add_argument("--num_spks", type=int, default=2)
    parser.add_argument("--preload_audio", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--engine_mode", type=str, default="train")
    parser.add_argument("--out_wav_dir", type=str, default=None)
    parser.add_argument(
        "--sepformer_config",
        type=str,
        default="base",
        choices=["base", "large"],
        help="Which SepFormer config to use: 'base' (smaller, for MiniLibriMix) or 'large' (original).",
    )
    args = parser.parse_args()

    if args.model == "sepformer":
        train_sepformer(args)
    else:
        train_mossformer2(args)


if __name__ == "__main__":
    main()
