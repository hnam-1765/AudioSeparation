"""
mir_eval stub — replaces mir_eval.separation.bss_eval_sources.
Computes: SDR, SIR, SAR (via BSB 3-term decomposition) + SI-SDR.

All pure numpy, no external dependencies.
"""
import numpy as np
from itertools import permutations


def l2norm(mat):
    return np.linalg.norm(mat, axis=-1)


def sisdr(ref, est, eps=1e-8):
    """Scale-invariant SDR (dB)."""
    ref_zm = ref - ref.mean(axis=-1, keepdims=True)
    est_zm = est - est.mean(axis=-1, keepdims=True)
    scale = (est_zm * ref_zm).sum(-1) / (ref_zm ** 2).sum(-1) + eps
    ref_s = scale[:, None] * ref_zm
    num = (ref_s ** 2).sum(-1)
    den = ((est_zm - ref_s) ** 2).sum(-1) + eps
    return 10 * np.log10((num + eps) / den)


def sdr(ref, est, eps=1e-8):
    """Standard SDR (dB)."""
    num = (ref ** 2).sum(-1)
    den = ((est - ref) ** 2).sum(-1) + eps
    return 10 * np.log10((num + eps) / den)


def bsb_decomposition(refs, ests, eps=1e-8):
    """
    BSS 3-term decomposition: SDR, SIR, SAR.

    refs, ests: list[K, (T,)] numpy (aligned permutation)
    Returns: SDR_i, SIR_i, SAR_i per source (dB)
    """
    K = len(refs)
    sdr_vals, sir_vals, sar_vals = [], [], []

    for k in range(K):
        r = refs[k].astype(np.float64)
        e = ests[k].astype(np.float64)

        # Target signal: optimal scaling of ref
        alpha = np.dot(e, r) / (np.dot(r, r) + eps)
        target = alpha * r

        # Interference: contribution from other references
        interference = np.zeros_like(r)
        for j in range(K):
            if j == k:
                continue
            alpha_j = np.dot(e, refs[j]) / (np.dot(refs[j], refs[j]) + eps)
            interference += alpha_j * refs[j]

        # Artifacts: residual after removing target
        artifacts = e - target

        def safe_db(num, den, eps=1e-8):
            return 10 * np.log10((num + eps) / (den + eps))

        sdr_k = safe_db(np.sum(target ** 2), np.sum((e - target) ** 2))
        sir_k = safe_db(np.sum(target ** 2), np.sum(interference ** 2))
        sar_k = safe_db(np.sum(target ** 2), np.sum(artifacts ** 2))

        sdr_vals.append(sdr_k)
        sir_vals.append(sir_k)
        sar_vals.append(sar_k)

    return np.array(sdr_vals), np.array(sir_vals), np.array(sar_vals)


def bss_eval_sources(ref, est, ask_isolated_if_speech_is_too_long=False):
    """
    Stub for mir_eval.separation.bss_eval_sources.

    ref: (K, T) numpy — K reference sources
    est: (K, T) numpy — K estimated sources (alignment may differ)

    Returns: (SDR, SIR, SAR, perm) — SDR/SIR/SAR are (K,) arrays (dB), perm is best ordering
    """
    K = ref.shape[0]

    # PIT: find best permutation over SI-SDR
    scores = []
    for perm in permutations(range(K)):
        vals = [sisdr(ref[perm[t]], est[t]) for t in range(K)]
        scores.append(np.mean(vals))
    best_perm = list(permutations(range(K)))[np.argmax(scores)]
    aligned_ests = [est[t] for t in best_perm]
    aligned_refs = [ref[t] for t in best_perm]

    # SDR, SIR, SAR via BSB decomposition
    sdr_vals, sir_vals, sar_vals = bsb_decomposition(aligned_refs, aligned_ests)

    return sdr_vals, sir_vals, sar_vals, best_perm