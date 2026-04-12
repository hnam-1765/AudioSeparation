import os
import random

import numpy as np
import scipy.io.wavfile as wav
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from sepformer_utils.decorators import logger_wraps
from sepformer_utils.util_dataset import parse_scps


def load_wav(path, sr=8000):
    """Load WAV using scipy."""
    rate, data = wav.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)
    if sr != rate:
        ratio = sr / rate
        new_len = int(len(data) * ratio)
        idx = np.linspace(0, len(data) - 1, new_len)
        data = np.interp(idx, np.arange(len(data)), data).astype(np.float32)
        sr = rate
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


@logger_wraps()
def get_dataloaders(args, dataset_config, loader_config):
    partitions = ["test"] if "test" in args.engine_mode else ["train", "valid", "test"]
    dataloaders = {}
    for partition in partitions:
        scp_mix = os.path.join(dataset_config["scp_dir"], dataset_config[partition]["mixture"])
        scp_srcs = [
            os.path.join(dataset_config["scp_dir"], dataset_config[partition][spk_key])
            for spk_key in dataset_config[partition]
            if spk_key.startswith("spk")
        ]
        dataset = MyDataset(
            max_len=dataset_config["max_len"],
            fs=dataset_config["sampling_rate"],
            partition=partition,
            wave_scp_srcs=scp_srcs,
            wave_scp_mix=scp_mix,
            dynamic_mixing=dataset_config[partition].get("dynamic_mixing", False) if partition == "train" else False,
        )
        dataloaders[partition] = DataLoader(
            dataset=dataset,
            batch_size=1 if partition == "test" else loader_config["batch_size"],
            shuffle=partition == "train",
            pin_memory=loader_config["pin_memory"],
            num_workers=loader_config["num_workers"],
            drop_last=loader_config["drop_last"],
            collate_fn=_collate,
        )
    return dataloaders


def _collate(egs):
    def _prepare_target(dict_list, index):
        return torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(d["src"][index], dtype=torch.float32) for d in dict_list],
            batch_first=True,
        )

    if not isinstance(egs, list):
        raise ValueError(f"Unsupported index type({type(egs)})")

    dict_list = sorted(egs, key=lambda x: x["num_sample"], reverse=True)
    num_spks = len(dict_list[0]["src"])
    mixture = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(d["mix"], dtype=torch.float32) for d in dict_list],
        batch_first=True,
    )
    src = [_prepare_target(dict_list, index) for index in range(num_spks)]
    input_sizes = torch.tensor([d["num_sample"] for d in dict_list], dtype=torch.float32)
    key = [d["key"] for d in dict_list]
    return input_sizes, mixture, src, key


@logger_wraps()
class MyDataset(Dataset):
    def __init__(self, max_len, fs, partition, wave_scp_srcs, wave_scp_mix, dynamic_mixing, speed_list=None):
        self.partition = partition
        self.max_len = max_len
        self.fs = fs
        self.dynamic_mixing = dynamic_mixing
        self.speed_list = speed_list or [0.9, 1.0, 1.1]

        for wave_scp_src in wave_scp_srcs:
            if not os.path.exists(wave_scp_src):
                raise FileNotFoundError(f"Could not find file {wave_scp_src}")
        if not os.path.exists(wave_scp_mix):
            raise FileNotFoundError(f"Could not find file {wave_scp_mix}")

        self.wave_dict_srcs = [parse_scps(wave_scp_src) for wave_scp_src in wave_scp_srcs]
        self.wave_dict_mix = parse_scps(wave_scp_mix)
        self.wave_keys = list(self.wave_dict_mix.keys())
        logger.info(f"Create MyDataset for {wave_scp_mix} with {len(self.wave_dict_mix)} utterances")

    def __len__(self):
        return len(self.wave_dict_mix)

    def __contains__(self, key):
        return key in self.wave_dict_mix

    def _dynamic_mixing(self, key):
        def _match_length(wav_data, len_data):
            leftover = len(wav_data) - len_data
            idx = random.randint(0, leftover)
            return wav_data[idx : idx + len_data]

        samps_src = []
        src_len = []
        while True:
            key_random = random.choice(list(self.wave_dict_srcs[0].keys()))
            tmp1 = key.split("_")[1][:3] != key_random.split("_")[3][:3]
            tmp2 = key.split("_")[3][:3] != key_random.split("_")[1][:3]
            if tmp1 and tmp2:
                break

        idx1, idx2 = (0, 1) if random.random() > 0.5 else (1, 0)
        files = [self.wave_dict_srcs[idx1][key], self.wave_dict_srcs[idx2][key_random]]

        for idx, file in enumerate(files):
            if not os.path.exists(file):
                raise FileNotFoundError(f"Input file {file} does not exist")
            samps_tmp, _ = load_wav(file, sr=self.fs)

            if idx == 0:
                ref_rms = np.sqrt(np.mean(np.square(samps_tmp)))
            curr_rms = np.sqrt(np.mean(np.square(samps_tmp)))

            norm_factor = ref_rms / curr_rms
            samps_tmp *= norm_factor
            gain = pow(10, -random.uniform(-5, 5) / 20)
            samps_src.append(gain * np.array(torch.tensor(samps_tmp)))
            src_len.append(len(samps_tmp))

        min_len = min(src_len)
        samps_src = [_match_length(source, min_len) for source in samps_src]
        samps_mix = sum(samps_src)

        if len(samps_mix) % 4 != 0:
            remains = len(samps_mix) % 4
            samps_mix = samps_mix[:-remains]
            samps_src = [source[:-remains] for source in samps_src]

        if self.partition != "test" and len(samps_mix) > self.max_len:
            start = random.randint(0, len(samps_mix) - self.max_len)
            samps_mix = samps_mix[start : start + self.max_len]
            samps_src = [source[start : start + self.max_len] for source in samps_src]
        return samps_mix, samps_src

    def _direct_load(self, key):
        samps_src = []
        files = [wave_dict_src[key] for wave_dict_src in self.wave_dict_srcs]
        for file in files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"Input file {file} does not exist")
            samps_tmp, _ = load_wav(file, sr=self.fs)
            samps_src.append(samps_tmp)

        file = self.wave_dict_mix[key]
        if not os.path.exists(file):
            raise FileNotFoundError(f"Input file {file} does not exist")
        samps_mix, _ = load_wav(file, sr=self.fs)

        if len(samps_mix) % 4 != 0:
            remains = len(samps_mix) % 4
            samps_mix = samps_mix[:-remains]
            samps_src = [source[:-remains] for source in samps_src]

        if self.partition != "test" and len(samps_mix) > self.max_len:
            start = random.randint(0, len(samps_mix) - self.max_len)
            samps_mix = samps_mix[start : start + self.max_len]
            samps_src = [source[start : start + self.max_len] for source in samps_src]

        return samps_mix, samps_src

    def __getitem__(self, index):
        key = self.wave_keys[index]
        if any(key not in source_dict for source_dict in self.wave_dict_srcs) or key not in self.wave_dict_mix:
            raise KeyError(f"Could not find utterance {key}")
        samps_mix, samps_src = self._dynamic_mixing(key) if self.dynamic_mixing else self._direct_load(key)
        return {"num_sample": samps_mix.shape[0], "mix": samps_mix, "src": samps_src, "key": key}
