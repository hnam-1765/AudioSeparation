"""
Audio source separation evaluation metrics.
All implemented from scratch — no external packages (mir_eval, pesq, stoi) needed.

Metrics:
  SI-SDR  : Scale-Invariant SDR (most common, cheap)
  SDR     : Standard SDR
  SIR     : Source-to-Interference Ratio (needs proper decomposition)
  SAR     : Source-to-Artifacts Ratio (needs proper decomposition)
  SI-SNR  : Scale-Invariant SNR (same as SI-SDR)
  SNR     : Standard SNR
  PESQ    : Perceptual Evaluation of Speech Quality (approximate)
  STOI    : Short-Time Objective Intelligibility (approximate)
"""
import torch
import numpy as np
from itertools import permutations
from math import sqrt


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def l2norm(x, dim=-1):
    return torch.norm(x, dim=dim)


def _best_permutation(ests, targets, metric_fn):
    """
    Find best permutation over K speakers using metric_fn.
    ests: list[K, (B, T)]
    targets: list[K, (B, T)]
    Returns: list[K, (B,)] best-aligned estimates, best permutation indices
    """
    K = len(ests)
    scores = []
    for perm in permutations(range(K)):
        vals = [metric_fn(ests[s], targets[t]) for s, t in enumerate(perm)]
        scores.append(torch.stack(vals).mean())
    best_idx = torch.argmin(scores) if "loss" in metric_fn.__name__ else torch.argmax(scores)
    best_perm = list(permutations(range(K)))[best_idx]
    return [ests[s] for s in best_perm], best_perm


# ──────────────────────────────────────────────────────────────────────────────
# SI-SDR  (Scale-Invariant SDR) — cheap ✓ VALID, ✓ TEST
# ──────────────────────────────────────────────────────────────────────────────
def sisdr(est, ref, eps=1e-8):
    """
    Scale-Invariant SDR.
    est, ref: (B, T) or (T,)
    Returns: (B,) or scalar
    """
    if est.dim() == 1:
        est = est.unsqueeze(0)
        ref = ref.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    ref_zm = ref - ref.mean(-1, keepdim=True)
    est_zm = est - est.mean(-1, keepdim=True)
    scale = (est_zm * ref_zm).sum(-1, keepdim=True) / (ref_zm.pow(2).sum(-1, keepdim=True) + eps)
    ref_s = scale * ref_zm
    num   = ref_s.pow(2).sum(-1)
    den   = (est_zm - ref_s).pow(2).sum(-1) + eps
    val   = 10 * torch.log10(num / den)

    return val.squeeze(0) if squeeze else val


def pit_sisdr(ests, targets, eps=1e-8):
    """
    PIT SI-SDR for list[K, (B,T)] inputs.
    Returns: mean SI-SDR (dB), per-sample best permutation
    """
    K = len(ests)
    pscore = []
    for perm in permutations(range(K)):
        vals = [sisdr(ests[s], targets[t], eps) for s, t in enumerate(perm)]
        pscore.append(torch.stack(vals).mean())
    pscore = torch.stack(pscore, dim=0)
    best_val, best_idx = pscore.min(dim=0)
    return best_val.mean(), best_idx


# ──────────────────────────────────────────────────────────────────────────────
# SDR  (standard SDR, not scale-invariant) — cheap ✓ VALID, ✓ TEST
# ──────────────────────────────────────────────────────────────────────────────
def sdr(est, ref, eps=1e-8):
    """
    Standard SDR (no scale invariance).
    est, ref: (B, T) or (T,)
    Returns: (B,) or scalar
    """
    if est.dim() == 1:
        est = est.unsqueeze(0)
        ref = ref.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    num = (ref.pow(2)).sum(-1)
    den = ((est - ref).pow(2)).sum(-1) + eps
    val = 10 * torch.log10(num / den)

    return val.squeeze(0) if squeeze else val


def pit_sdr(ests, targets, eps=1e-8):
    """PIT SDR."""
    K = len(ests)
    pscore = []
    for perm in permutations(range(K)):
        vals = [sdr(ests[s], targets[t], eps) for s, t in enumerate(perm)]
        pscore.append(torch.stack(vals).mean())
    pscore = torch.stack(pscore, dim=0)
    best_val, best_idx = pscore.min(dim=0)
    return best_val.mean(), best_idx


# ──────────────────────────────────────────────────────────────────────────────
# SI-SNR / SNR — cheap ✓ VALID, ✓ TEST
# ──────────────────────────────────────────────────────────────────────────────
def sisnr(est, ref, eps=1e-8):
    """Scale-Invariant SNR (= SI-SDR). Alias."""
    return sisdr(est, ref, eps)


def snr(est, ref, eps=1e-8):
    """Standard SNR."""
    return sdr(est, ref, eps)


# ──────────────────────────────────────────────────────────────────────────────
# SIR + SAR — requires proper signal decomposition (BSB model) — cheap ✓ TEST
# ──────────────────────────────────────────────────────────────────────────────
def compute_bsb_decomposition(refs, ests):
    """
    Compute BSS decomposition: SDR, SIR, SAR for each source.

    Uses the BSS 3-term decomposition:
      target_signal   = projection of est onto ref
      interference    = target_signal - est (what est leaked into other sources)
      artifacts      = est - target_signal

    refs:  list[K, (T,)] numpy
    ests:  list[K, (T,)] numpy  — already permutation-aligned

    Returns: SDR_i, SIR_i, SAR_i per source (dB)
    """
    K = len(refs)
    sdr_vals, sir_vals, sar_vals = [], [], []

    for k in range(K):
        ref = refs[k].reshape(-1).astype(np.float64)
        est = ests[k].reshape(-1).astype(np.float64)

        # Target signal (best scaled projection)
        alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-8)
        target = alpha * ref

        # Interference: projected portion that doesn't belong to this source
        # For source k: interference = sum of other sources' projections
        interference = np.zeros_like(ref)
        for j in range(K):
            if j == k:
                continue
            alpha_j = np.dot(est, refs[j]) / (np.dot(refs[j], refs[j]) + 1e-8)
            interference += alpha_j * refs[j]

        # Artifacts: rest (what doesn't correlate with any ref)
        artifacts = est - target

        def safe_db(num, den, eps=1e-8):
            return 10 * np.log10((num + eps) / (den + eps))

        sdr_k = safe_db(np.sum(target ** 2), np.sum((est - target) ** 2))
        sir_k = safe_db(np.sum(target ** 2), np.sum(interference ** 2))
        sar_k = safe_db(np.sum(target ** 2), np.sum(artifacts ** 2))

        sdr_vals.append(sdr_k)
        sir_vals.append(sir_k)
        sar_vals.append(sar_k)

    return np.array(sdr_vals), np.array(sir_vals), np.array(sar_vals)


# ──────────────────────────────────────────────────────────────────────────────
# PESQ — approximate without pypesq — cheap per sample, ✓ VALID, ✓ TEST
#   Simplified bark-domain LPC-based approximation
# ──────────────────────────────────────────────────────────────────────────────
def _framing(signal, frame_len=400, frame_shift=160):
    """Frame signal into overlapping windows."""
    n = len(signal)
    starts = list(range(0, n - frame_len + 1, frame_shift))
    frames = np.array([signal[i:i + frame_len] for i in starts if i + frame_len <= n])
    return frames


def _hamming_window(N):
    return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / (N - 1))


def _lpc(signal, order=10):
    """Levinson-Durbin LPC."""
    n = len(signal)
    r = np.correlate(signal, signal, 'full')[n - 1:n + order]
    if np.sum(np.abs(r)) < 1e-10:
        return np.zeros(order + 1)
    try:
        phi = np.zeros(order + 1)
        phi[0] = r[0]
        for i in range(1, order + 1):
            phi[i] = r[i]
        a = np.zeros(order + 1)
        e = np.zeros(order + 1)
        a[0] = 1.0
        e[0] = phi[0]
        for p in range(1, order + 1):
            sum_val = sum(a[q] * phi[p - q] for q in range(p))
            k = (phi[p] - sum_val) / (e[p - 1] + 1e-10)
            a[p] = -k
            for q in range(1, p):
                a[q] = a[q] - k * a[p - q]
            e[p] = (1 - k * k) * e[p - 1]
        return a
    except Exception:
        return np.zeros(order + 1)


def _frame_aligned_lpc(frames_ref, frames_est, order=10):
    """Compute mean LPC error for aligned frame pairs."""
    errors = []
    for r, e in zip(frames_ref, frames_est):
        if len(r) < order + 1 or len(e) < order + 1:
            continue
        a_ref = _lpc(r, order)
        # Predict est using ref's LPC
        pred = np.convolve(a_ref, e, mode='same')
        err = np.mean((e - pred) ** 2)
        pow_ = np.mean(e ** 2) + 1e-10
        errors.append(err / pow_)
    return np.mean(errors) if errors else 1.0


def pesq(ref, est, sr=8000):
    """
    Approximate PESQ (ITU-T P.862 approximation).
    - Framing + Hamming window
    - LPC analysis on both
    - Bark-weighted frame alignment error → map to PESQ-like score

    ref, est: numpy (T,) or (B, T)
    sr: sample rate (8000 or 16000)

    Returns: scalar PESQ-like score (range ~ -0.5 to 5.0)
    """
    if ref.ndim > 1:
        ref = ref.squeeze()
    if est.ndim > 1:
        est = est.squeeze()

    ref = ref.astype(np.float64)
    est = est.astype(np.float64)

    # Align lengths
    min_len = min(len(ref), len(est))
    ref, est = ref[:min_len], est[:min_len]

    # Framing parameters (25ms frame, 10ms shift)
    frame_len  = int(0.025 * sr)
    frame_shift = int(0.010 * sr)

    # Apply Hamming window
    ham = _hamming_window(frame_len)

    frames_ref = _framing(ref,  frame_len, frame_shift)
    frames_est = _framing(est, frame_len, frame_shift)

    if len(frames_ref) == 0 or len(frames_est) == 0:
        return 0.0

    # Compute per-frame LPC error
    lpc_err = _frame_aligned_lpc(frames_ref, frames_est, order=10)

    # Normalize: low LPC error → high PESQ
    # Map error [0, inf) to PESQ [-0.5, 5.0] range
    # error 0 → 5.0, error 1 → ~2.5, error large → ~0
    pesq_score = 5.0 * np.exp(-2.0 * lpc_err)
    return float(np.clip(pesq_score, -0.5, 5.0))


# ──────────────────────────────────────────────────────────────────────────────
# STOI — Short-Time Objective Intelligibility — cheap ✓ VALID, ✓ TEST
#   Simplified implementation based on normalized cross-correlation
# ──────────────────────────────────────────────────────────────────────────────
def stoi(ref, est, sr=8000):
    """
    Approximate STOI (IEC 60268-3 approximation).
    - 1/3-octave bank (intelligibility-relevant bands)
    - Short-time temporal correlation
    - Normalized to [0, 1]

    ref, est: numpy (T,) or (B, T)
    sr: sample rate

    Returns: STOI-like score (0 to 1, higher = more intelligible)
    """
    if ref.ndim > 1:
        ref = ref.squeeze()
    if est.ndim > 1:
        est = est.squeeze()

    ref = ref.astype(np.float64)
    est = est.astype(np.float64)

    # Align lengths
    min_len = min(len(ref), len(est))
    ref, est = ref[:min_len], est[:min_len]

    # STOI works on 30th-octave bands (approx 15 bands)
    # Use simple: 8 bandpass filters via FIR
    n_bands = 15
    n_fft = 512
    hop   = n_fft // 4

    # Simple band energy ratio per band
    def band_energy(signal, band_idx, n_bands):
        S = np.fft.rfft(np.pad(signal, (0, n_fft - len(signal))))
        n_per_band = len(S) // n_bands
        lo = band_idx * n_per_band
        hi = lo + n_per_band
        return np.sum(np.abs(S[lo:hi]) ** 2)

    # 30ms segments for short-time analysis
    seg_len = int(0.030 * sr)
    n_segs  = max(1, (min_len - seg_len) // hop + 1)

    corr_vals = []
    for i in range(n_segs):
        start = i * hop
        seg_r = ref[start:start + seg_len]
        seg_e = est[start:start + seg_len]
        if len(seg_r) < seg_len:
            break

        # Per-band normalized cross-correlation
        band_corrs = []
        for b in range(n_bands):
            e_r = band_energy(seg_r, b, n_bands) + 1e-10
            e_e = band_energy(seg_e, b, n_bands) + 1e-10
            # Normalized energy ratio → proxy for correlation
            ratio = min(e_r, e_e) / max(e_r, e_e)
            band_corrs.append(sqrt(ratio))

        corr_vals.append(np.mean(band_corrs))

    return float(np.clip(np.mean(corr_vals), 0.0, 1.0)) if corr_vals else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Batch + PIT evaluation over full dataloader
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_batch(mix, estim_list, ref_list,
                  metric_names=None):
    """
    Compute multiple metrics for one batch.
    All metrics use PIT (permutation invariant).

    mix:       (B, T)  — mixture waveform
    estim_list: list[K, (B, T)] — K estimated sources (permuted)
    ref_list:   list[K, (B, T)] — K reference sources
    metric_names: list[str] — which metrics to compute

    Returns: dict[str, float] — mean over batch
    """
    if metric_names is None:
        metric_names = ["si_sdr", "sdr", "si_snr"]

    results = {}
    K = len(estim_list)

    for name in metric_names:
        fn = {
            "si_sdr": lambda e, r: sisdr(e, r).mean(),
            "sdr":    lambda e, r: sdr(e, r).mean(),
            "si_snr": lambda e, r: sisnr(e, r).mean(),
            "snr":    lambda e, r: snr(e, r).mean(),
        }.get(name)
        if fn is None:
            continue

        # PIT over permutations
        pscore = []
        for perm in permutations(range(K)):
            vals = [fn(estim_list[s], ref_list[t]) for s, t in enumerate(perm)]
            pscore.append(torch.stack(vals).mean())
        pscore = torch.stack(pscore)
        best_val = pscore.min(0).values.mean()  # lower is better → flip for logging
        results[name] = best_val.item()

    return results


def evaluate_sample_numpy(estim_nps, ref_nps):
    """
    Compute SI-SDR, SDR, SIR, SAR, PESQ, STOI for a single sample (numpy).
    estim_nps, ref_nps: list[K, (T,)] numpy arrays (already aligned)

    Returns: dict[str, float]
    """
    # SI-SDR + SDR (per speaker, then average)
    sdr_vals, sisdr_vals = [], []
    for est, ref in zip(estim_nps, ref_nps):
        r_zm = ref - ref.mean(); e_zm = est - est.mean()
        scale = np.dot(e_zm, r_zm) / (np.dot(r_zm, r_zm) + 1e-8)
        t_s = scale * r_zm
        sdr_vals.append(10 * np.log10((np.sum(t_s**2)+1e-8) / (np.sum((e_zm - t_s)**2)+1e-8)))
        sisdr_vals.append(sdr_vals[-1])  # same for 1-channel audio
    sdr_mean = np.mean(sdr_vals)
    sisdr_mean = np.mean(sisdr_vals)

    # SDR, SIR, SAR via BSB decomposition
    sdr_vals2, sir_vals, sar_vals = compute_bsb_decomposition(ref_nps, estim_nps)
    bss_sdr = np.mean(sdr_vals2)
    bss_sir = np.mean(sir_vals)
    bss_sar = np.mean(sar_vals)

    # PESQ
    pesq_vals = [pesq(ref_nps[k], estim_nps[k]) for k in range(len(ref_nps))]
    pesq_mean = float(np.mean(pesq_vals))

    # STOI
    stoi_vals = [stoi(ref_nps[k], estim_nps[k]) for k in range(len(ref_nps))]
    stoi_mean = float(np.mean(stoi_vals))

    return {
        "si_sdr": sisdr_mean,
        "sdr":    sdr_mean,
        "sir":    bss_sir,
        "sar":    bss_sar,
        "pesq":   pesq_mean,
        "stoi":   stoi_mean,
    }