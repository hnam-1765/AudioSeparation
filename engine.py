"""
Training engine for audio source separation models.
Supports both SepFormer and MossFormer2.
"""
import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.losses import pit_sisnr, sisnri_loss, pit_sisnr_mag, STFT
from utils.train_utils import save_checkpoint, load_checkpoint, get_last_checkpoint_path


class SeparationEngine:
    """
    Generic training/validation engine for audio source separation.
    Works with any model that returns:
      - SepFormer: (spk_audio, audio_aux)  — tuple
      - MossFormer2: (B, T, num_spks) tensor
    """
    def __init__(self, model, optimizer, config, device='cuda', log_dir="./logs"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.num_spks = config.get('num_spks', 2)
        self.max_epochs = config.get('max_epochs', 100)
        self.clip_grad = config.get('clip_grad', 5.0)
        self.mvn = config.get('mvn', False)
        self.use_aux_loss = config.get('use_aux_loss', False)
        self.frame_length = config.get('frame_length', 512)

        # Loss: STFT for magnitude-domain PIT
        self.stft_loss = STFT(frame_len=self.frame_length,
                               frame_hop=self.frame_length // 4).to(device)

        # Checkpoint state
        self.start_epoch = 0
        self.best_loss = float('inf')

        # LR scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8,
            patience=3, min_lr=1e-7)

        # Warmup scheduler
        self.warmup_steps = config.get('warmup_steps', 1000)
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.0, end_factor=1.0,
            total_iters=self.warmup_steps)

    # ── Training step ────────────────────────────────────────────────────────
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss, total_sisnr, n_batches = 0.0, 0.0, 0

        pbar = tqdm(dataloader, desc=f"[Train] Epoch {epoch}", dynamic_ncols=True)
        for mix, srcs, lengths, keys in pbar:
            mix = mix.to(self.device)
            srcs = [s.to(self.device) for s in srcs]

            # Warmup on epoch 1
            if epoch == 1:
                self.warmup_scheduler.step()

            self.optimizer.zero_grad()

            # Forward
            outputs = self.model(mix)

            # Normalise output shape
            if isinstance(outputs, tuple):
                spk_audio, audio_aux = outputs
            else:
                spk_audio = [outputs[:, :, k] for k in range(self.num_spks)]
                audio_aux = []

            # Main SI-SNR loss
            loss_sisnr, _ = pit_sisnr(spk_audio, srcs)
            total_sisnr += loss_sisnr.item()
            loss = loss_sisnr

            # Auxiliary freq-domain losses (SepFormer only)
            if self.use_aux_loss and audio_aux:
                alpha = 0.4 * (0.8 ** max(0, (epoch - 101) // 5)) if epoch > 100 else 0.4
                for stage_out in audio_aux:
                    stage_spks = [stage_out[k] for k in range(self.num_spks)]
                    loss_freq = pit_sisnr_mag(stage_spks, srcs, lengths, self.stft_loss)
                    loss = loss + alpha * loss_freq
                loss = loss / self.num_spks

            # Backward + step
            loss.backward()
            if self.clip_grad:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({
                'sisnr': f'{loss_sisnr.item():.2f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })

        pbar.close()
        return total_loss / n_batches, total_sisnr / n_batches

    # ── Validation step ──────────────────────────────────────────────────────
    def valid_epoch(self, dataloader, epoch):
        self.model.eval()
        total_loss, total_sisnr, n_batches = 0.0, 0.0, 0

        with torch.inference_mode():
            pbar = tqdm(dataloader, desc=f"[Valid] Epoch {epoch}", dynamic_ncols=True)
            for mix, srcs, lengths, keys in pbar:
                mix = mix.to(self.device)
                srcs = [s.to(self.device) for s in srcs]

                outputs = self.model(mix)
                if isinstance(outputs, tuple):
                    spk_audio, _ = outputs
                else:
                    spk_audio = [outputs[:, :, k] for k in range(self.num_spks)]

                loss_sisnr, _ = pit_sisnr(spk_audio, srcs)
                total_loss += loss_sisnr.item()
                total_sisnr += loss_sisnr.item()
                n_batches += 1
                pbar.set_postfix({'sisnr': f'{loss_sisnr.item():.2f}'})

            pbar.close()
        return total_loss / n_batches, total_sisnr / n_batches

    # ── Full training loop ──────────────────────────────────────────────────
    def run(self, train_loader, valid_loader, resume=True):
        if resume:
            ckpt_path = get_last_checkpoint_path(self.log_dir)
            if ckpt_path:
                self.start_epoch, self.best_loss = load_checkpoint(
                    ckpt_path, self.model, self.optimizer, self.device)
                print(f"Resuming from epoch {self.start_epoch}")

        for epoch in range(self.start_epoch + 1, self.max_epochs + 1):
            t0 = time.time()

            train_loss, train_sisnr = self.train_epoch(train_loader, epoch)
            t_train = time.time() - t0

            valid_loss, valid_sisnr = self.valid_epoch(valid_loader, epoch)
            t_valid = time.time() - t0 - t_train

            if epoch > self.config.get('start_scheduling', 10):
                self.scheduler.step(valid_loss)

            lr = self.optimizer.param_groups[0]['lr']
            print(
                f"[Epoch {epoch:3d}] "
                f"TrainSISNR={train_sisnr:7.2f}dB | "
                f"ValidSISNR={valid_sisnr:7.2f}dB | "
                f"LR={lr:.2e} | "
                f"t_train={t_train:.1f}s t_valid={t_valid:.1f}s"
            )

            # Checkpoint
            is_best = valid_loss < self.best_loss
            if is_best:
                self.best_loss = valid_loss
            latest_ckpt_path = os.path.join(self.log_dir, "latest.pt")
            save_checkpoint(epoch, self.model, self.optimizer, latest_ckpt_path,
                            best_loss=self.best_loss)
            if is_best:
                best_ckpt_path = os.path.join(self.log_dir, "best.pt")
                save_checkpoint(epoch, self.model, self.optimizer, best_ckpt_path,
                                best_loss=self.best_loss)

        print(f"\nTraining done! Best valid loss: {self.best_loss:.4f}")

    # ── Inference helper ─────────────────────────────────────────────────────
    @torch.no_grad()
    def inference(self, mix_wav):
        """Run inference on a single waveform tensor (B,T) or (T,)."""
        self.model.eval()
        if mix_wav.dim() == 1:
            mix_wav = mix_wav.unsqueeze(0)
        mix_wav = mix_wav.to(self.device)
        outputs = self.model(mix_wav)
        if isinstance(outputs, tuple):
            return outputs[0]   # list[K, (B,T)]
        return [outputs[:, :, k] for k in range(self.num_spks)]
