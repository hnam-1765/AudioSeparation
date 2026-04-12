"""
Audio utilities: loading WAV files using scipy, STFT/ISTFT, pad/trim helpers.
All audio is 8kHz mono (int16 PCM → float32 [-1, 1]).
"""
import torch
import numpy as np
import scipy.io.wavfile as wav


def load_wav(path: str, target_sr: int = 8000) -> tuple[np.ndarray, int]:
    """Load WAV file using scipy (works without librosa/soundfile)."""
    sr, data = wav.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)
    if sr != target_sr:
        # Simple linear resampling
        ratio = target_sr / sr
        new_len = int(len(data) * ratio)
        indices = np.linspace(0, len(data) - 1, new_len)
        data = np.interp(indices, np.arange(len(data)), data).astype(np.float32)
        sr = target_sr
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def save_wav(path: str, audio: np.ndarray, sr: int = 8000):
    """Save WAV file using scipy."""
    audio = np.clip(audio, -1.0, 1.0)
    data = (audio * 32767).astype(np.int16)
    wav.write(path, sr, data)


def pad_audio(audio: np.ndarray, target_len: int) -> np.ndarray:
    """Pad audio to target_len with zeros."""
    if len(audio) < target_len:
        pad_len = target_len - len(audio)
        audio = np.pad(audio, (0, pad_len), mode='constant')
    return audio


def trim_audio(audio: np.ndarray, target_len: int) -> np.ndarray:
    """Trim audio to target_len."""
    return audio[:target_len]


def collate_variable_length(batch):
    """
    Collate batch with variable-length audio.
    batch: list of dicts with 'mix', 'srcs', 'key', 'num_sample'
    Returns: (padded_mix BxT, list of K padded_src BxT, lengths B)
    """
    # Sort descending by length
    batch = sorted(batch, key=lambda x: x['num_sample'], reverse=True)
    mix_list = [torch.from_numpy(b['mix']).float() for b in batch]
    src_lists = [b['srcs'] for b in batch]
    lengths = torch.tensor([b['num_sample'] for b in batch], dtype=torch.long)
    keys = [b['key'] for b in batch]

    # Pad mixture
    mix_padded = torch.nn.utils.rnn.pad_sequence(mix_list, batch_first=True)

    # Pad each source
    K = len(src_lists[0])
    src_padded = []
    for k in range(K):
        src_k = [torch.from_numpy(s[k]).float() for s in src_lists]
        src_padded.append(torch.nn.utils.rnn.pad_sequence(src_k, batch_first=True))

    return mix_padded, src_padded, lengths, keys
