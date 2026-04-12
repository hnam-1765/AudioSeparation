"""
PIT (Permutation Invariant Training) losses for audio source separation.
Implements: SI-SNR, SI-SNRi, SISNR-magnitude (frequency-domain).
"""
import torch
import math
import numpy as np
from itertools import permutations


def l2_norm(mat, keepdim=False):
    return torch.norm(mat, dim=-1, keepdim=keepdim)


def sisnr_loss(est, tgt, eps=1e-8):
    """
    Scale-Invariant SNR between est and tgt.
    est, tgt: (B, T)
    Returns: (B,)
    """
    tgt_zm = tgt - tgt.mean(dim=-1, keepdim=True)
    est_zm = est - est.mean(dim=-1, keepdim=True)
    scale = torch.sum(est_zm * tgt_zm, dim=-1, keepdim=True) / \
            (l2_norm(tgt_zm, True)**2 + eps)
    tgt_s = torch.clamp(scale, min=1e-2) * tgt_zm
    loss = -20 * torch.log10(eps + l2_norm(tgt_s) / (l2_norm(est_zm - tgt_s) + eps))
    return torch.clamp(loss, min=-30)


def pit_sisnr(estims, targets, eps=1e-8):
    """
    PIT SI-SNR over permutation.
    estims: list of K tensors (B, T)
    targets: list of K tensors (B, T)
    Returns: mean loss (scalar), best permutation indices per sample
    """
    K = len(estims)
    if K == 1:
        return sisnr_loss(estims[0], targets[0], eps).mean()

    pscore = []
    for perm in permutations(range(K)):
        losses = [sisnr_loss(estims[s], targets[t], eps) for s, t in enumerate(perm)]
        pscore.append(sum(losses) / K)
    pscore = torch.stack(pscore, dim=0)  # (num_perms, B)
    min_perutt, best_perm = torch.min(pscore, dim=0)
    return min_perutt.mean(), best_perm


def sisnri_loss(estims, mixtures, targets, eps=1e-8):
    """
    SI-SNR Improvement: SDR(est, tgt) - SDR(mix, tgt)
    estims: list[K, (B,T)]
    mixtures: (B,T) raw mixture
    targets: list[K, (B,T)]
    Returns: mean SISNRi, per-source SISNRi list
    """
    K = len(estims)
    mix_zm = mixtures - mixtures.mean(dim=-1, keepdim=True)

    pscore = []
    for perm in permutations(range(K)):
        sisnr_imp = []
        for s, t in enumerate(perm):
            tgt = targets[t]
            tgt_zm = tgt - tgt.mean(dim=-1, keepdim=True)
            est = estims[s]
            est_zm = est - est.mean(dim=-1, keepdim=True)

            # SDR(est, tgt)
            scale_e = torch.sum(est_zm * tgt_zm, dim=-1) / (l2_norm(tgt_zm)**2 + eps)
            tgt_s_e = torch.clamp(scale_e, min=1e-2) * tgt_zm
            sdr_e = 20 * torch.log10(eps + l2_norm(tgt_s_e) / (l2_norm(est_zm - tgt_s_e) + eps))

            # SDR(mix, tgt)
            scale_m = torch.sum(mix_zm * tgt_zm, dim=-1) / (l2_norm(tgt_zm)**2 + eps)
            tgt_s_m = torch.clamp(scale_m, min=1e-2) * tgt_zm
            sdr_m = 20 * torch.log10(eps + l2_norm(tgt_s_m) / (l2_norm(mix_zm - tgt_s_m) + eps))

            sisnr_imp.append(sdr_e - sdr_m)
        pscore.append(torch.stack(sisnr_imp))
    pscore = torch.stack(pscore, dim=0)  # (num_perms, K, B)
    best_idx = torch.argmax(pscore.sum(1), dim=0)  # (B,)
    best = pscore[best_idx, torch.arange(best_idx.shape[0])]  # (K,)
    return best.sum(0).mean(), [b.mean().item() for b in best]


# ──────────────────────────────────────────────────────────────────────────────
# STFT / Magnitude-loss helper (no torchaudio/librosa needed)
# ──────────────────────────────────────────────────────────────────────────────
class STFT(torch.nn.Module):
    """
    Simple STFT using rfft on Hamming window.
    Works with (B, T) waveform → (B, F, Frames) magnitude + phase.
    """
    def __init__(self, frame_len=512, frame_hop=128, window='hann'):
        super().__init__()
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        if window == 'hann':
            win = torch.hann_window(frame_len)
        else:
            win = torch.ones(frame_len)
        # Normalize for perfect reconstruction
        const = (2 / 3) ** 0.5
        win = const * win
        self.register_buffer('window', win)

    def forward(self, x, cplx=False):
        """
        x: (B, T) or (T,)
        Returns: mag (B, F, Frames), phase (B, F, Frames)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B, T = x.shape
        hop = self.frame_hop
        frames = (T - self.frame_len) // hop + 1
        # pad if needed
        pad_len = frames * hop + self.frame_len - self.frame_len
        if pad_len > T:
            pad_len = frames * hop + self.frame_len - self.frame_len
            x = torch.nn.functional.pad(x, (0, pad_len))
        # segment into overlapping frames
        x = x.unfold(1, self.frame_len, hop)  # (B, Frames, frame_len)
        x = x * self.window  # (B, Frames, frame_len)
        # STFT
        X = torch.fft.rfft(x, dim=2)  # (B, Frames, F)
        mag = torch.abs(X)
        phase = torch.atan2(X.imag, X.real)
        if cplx:
            return mag, phase
        return mag


def pit_sisnr_mag(estims, targets, input_sizes, stft_model, eps=1e-12):
    """
    PIT SI-SNR in magnitude spectral domain.
    estims: list[K, (B,T)]
    targets: list[K, (B,T)]
    stft_model: STFT instance
    """
    K = len(estims)
    pscore = []
    for perm in permutations(range(K)):
        losses = []
        for s, t in enumerate(perm):
            mix_mag = stft_model(estims[s])  # (B,F,T')
            src_mag = stft_model(targets[t])
            utt_loss = -20 * torch.log10(
                eps + l2_norm(l2_norm(src_mag)) / (l2_norm(mix_mag - src_mag) + eps))
            losses.append(utt_loss.mean())
        pscore.append(sum(losses) / K)
    pscore = torch.stack(pscore, dim=0)
    min_loss, _ = torch.min(pscore, dim=0)
    return min_loss.mean()
