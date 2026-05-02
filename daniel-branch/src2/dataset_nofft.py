import os
import glob
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

from config import Config
from dataset import discover_records, PacketRecord, load_waveform, \
                    normalize, apply_phase_augment, compute_pos_weight


class LSIGDatasetNoFFT(Dataset):
    """
    消融实验Dataset：不做FFT，直接用时域波形
    输出x shape: [2, n_symbols * n_fft] = [2, 768]
    对比LSIGDataset的 [2, 12, 64]
    """
    def __init__(
        self,
        records: List[PacketRecord],
        cfg: Config,
        augment: bool = False,
    ):
        self.records = records
        self.cfg = cfg
        self.augment = augment
        self.length = cfg.n_symbols * cfg.n_fft  # 768

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        wave = load_waveform(rec.bin_path)

        if self.augment and self.cfg.use_phase_augment:
            wave = apply_phase_augment(wave)

        wave = normalize(wave)

        # 直接取前768个点，不做FFT
        length = self.length
        if len(wave) >= length:
            wave = wave[:length]
        else:
            pad = np.zeros(length, dtype=np.complex64)
            pad[:len(wave)] = wave
            wave = pad

        # 分成real/imag两个channel，shape: [2, 768]
        x = np.stack([wave.real, wave.imag], axis=0).astype(np.float32)
        y = rec.lsig_bits.astype(np.float32)

        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
            "packet_id": rec.packet_id,
        }


def build_datasets_nofft(cfg: Config) -> Tuple:
    print("Discovering records (No-FFT)...")
    train_all = discover_records(cfg.root_dir, "Train")
    test_recs = discover_records(cfg.root_dir, "Test")

    def filter_records(recs):
        return [
            r for r in recs
            if r.protocol == cfg.target_protocol
            and r.lsig_bits is not None
            and len(r.lsig_bits) == 24
        ]

    train_all = filter_records(train_all)
    test_recs = filter_records(test_recs)

    train_recs, val_recs = train_test_split(
        train_all,
        test_size=cfg.val_split,
        random_state=cfg.random_seed,
    )

    print(f"Train: {len(train_recs)}, Val: {len(val_recs)}, Test: {len(test_recs)}")

    ds_train = LSIGDatasetNoFFT(train_recs, cfg, augment=True)
    ds_val   = LSIGDatasetNoFFT(val_recs,   cfg, augment=False)
    ds_test  = LSIGDatasetNoFFT(test_recs,  cfg, augment=False)

    return ds_train, ds_val, ds_test, train_recs