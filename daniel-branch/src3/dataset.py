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


# ─── Record ─────────────────────────────────────────────
@dataclass
class PacketRecord:
    packet_id: str
    bin_path: str      # Low/*.bin  (5 MHz sub-Nyquist — model input)
    mat_path: str      # Label/*.mat (decoded from High by MATLAB — ground truth)
    protocol: str
    lsig_bits: Optional[np.ndarray]   # 24-bit ground truth


# ─── File discovery ─────────────────────────────────────
def discover_records(root_dir: str, split: str) -> List[PacketRecord]:
    split_dir = os.path.join(root_dir, split)
    records = []

    for exp_dir in sorted(os.listdir(split_dir)):
        if exp_dir.startswith("_"):
            continue
        exp_path  = os.path.join(split_dir, exp_dir)
        low_dir   = os.path.join(exp_path, "Low")
        label_dir = os.path.join(exp_path, "Label")

        if not os.path.isdir(low_dir) or not os.path.isdir(label_dir):
            continue

        for bin_path in sorted(glob.glob(os.path.join(low_dir, "Packet_*.bin"))):
            idx      = os.path.basename(bin_path).replace("Packet_", "").replace(".bin", "")
            mat_path = os.path.join(label_dir, f"Packet_{idx}.mat")
            if not os.path.exists(mat_path):
                continue

            try:
                mat  = loadmat(mat_path)
                prot = str(np.squeeze(mat["prot"])).strip()
                lsig = np.asarray(mat["LSIGBit"]).reshape(-1).astype(np.float32) \
                       if "LSIGBit" in mat else None
            except Exception:
                continue

            records.append(PacketRecord(
                packet_id=f"{split}/{exp_dir}/Packet_{idx}",
                bin_path=bin_path,
                mat_path=mat_path,
                protocol=prot,
                lsig_bits=lsig,
            ))

    return records


# ─── Signal processing ──────────────────────────────────
def load_waveform(bin_path: str) -> np.ndarray:
    """Load Low/*.bin as complex64 IQ samples (5 MHz sub-Nyquist)."""
    return np.fromfile(bin_path, dtype=np.complex64)


def normalize(wave: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-std normalization (DeepMon Step 1)."""
    wave = wave - wave.mean()
    std  = wave.std()
    if std > 1e-8:
        wave = wave / std
    return wave


def apply_phase_augment(wave: np.ndarray) -> np.ndarray:
    """Random constant phase shift — simulates wireless channel phase distortion."""
    phase = np.random.uniform(0, 2 * np.pi)
    return wave * np.exp(1j * phase)


def wave_to_2d_spectrogram(
    wave: np.ndarray,
    n_fft: int = 64,
    n_symbols: int = 12,
) -> np.ndarray:
    """
    DeepMon Step 2: per-OFDM-symbol FFT (no cyclic prefix removal).
    Input : complex waveform [T]
    Output: [2, n_symbols, n_fft]  — real and imag as two channels
    """
    specs = []
    for i in range(n_symbols):
        start = i * n_fft
        end   = start + n_fft
        if start >= len(wave):
            seg = np.zeros(n_fft, dtype=np.complex64)
        elif end > len(wave):
            seg = np.zeros(n_fft, dtype=np.complex64)
            seg[:len(wave) - start] = wave[start:]
        else:
            seg = wave[start:end]
        specs.append(np.fft.fft(seg))

    spec = np.stack(specs, axis=0)                     # [n_symbols, n_fft]
    out  = np.stack([spec.real, spec.imag], axis=0)    # [2, n_symbols, n_fft]
    return out.astype(np.float32)


# ─── Dataset ────────────────────────────────────────────
class LSIGDataset(Dataset):
    def __init__(self, records: List[PacketRecord], cfg: Config, augment: bool = False):
        self.records = records
        self.cfg     = cfg
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec  = self.records[idx]
        wave = load_waveform(rec.bin_path)

        if self.augment and self.cfg.use_phase_augment:
            wave = apply_phase_augment(wave)

        wave = normalize(wave)
        x    = wave_to_2d_spectrogram(wave, self.cfg.n_fft, self.cfg.n_symbols)
        y    = rec.lsig_bits.astype(np.float32)

        return {
            "x":         torch.tensor(x, dtype=torch.float32),
            "y":         torch.tensor(y, dtype=torch.float32),
            "packet_id": rec.packet_id,
        }


# ─── No-FFT Dataset (ablation) ──────────────────────────
class LSIGDatasetNoFFT(Dataset):
    """
    Ablation: skip FFT, feed raw time-domain IQ directly.
    Output shape: [2, n_symbols * n_fft] = [2, 768]
    """
    def __init__(self, records: List[PacketRecord], cfg: Config, augment: bool = False):
        self.records = records
        self.cfg     = cfg
        self.augment = augment
        self.length  = cfg.n_symbols * cfg.n_fft   # 768

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec  = self.records[idx]
        wave = load_waveform(rec.bin_path)

        if self.augment and self.cfg.use_phase_augment:
            wave = apply_phase_augment(wave)

        wave = normalize(wave)

        L = self.length
        if len(wave) >= L:
            wave = wave[:L]
        else:
            pad = np.zeros(L, dtype=np.complex64)
            pad[:len(wave)] = wave
            wave = pad

        x = np.stack([wave.real, wave.imag], axis=0).astype(np.float32)
        y = rec.lsig_bits.astype(np.float32)

        return {
            "x":         torch.tensor(x, dtype=torch.float32),
            "y":         torch.tensor(y, dtype=torch.float32),
            "packet_id": rec.packet_id,
        }


# ─── Dataset builders ───────────────────────────────────
def _filter(recs: List[PacketRecord], cfg: Config) -> List[PacketRecord]:
    return [
        r for r in recs
        if r.protocol == cfg.target_protocol
        and r.lsig_bits is not None
        and len(r.lsig_bits) == 24
    ]


def build_datasets(cfg: Config, no_fft: bool = False) -> Tuple:
    print("Discovering records...")
    train_all = discover_records(cfg.root_dir, "Train")
    test_recs = discover_records(cfg.root_dir, "Test")

    train_all = _filter(train_all, cfg)
    test_recs = _filter(test_recs, cfg)

    print(f"Train (before split): {len(train_all)}")
    print(f"Test:                 {len(test_recs)}")

    train_recs, val_recs = train_test_split(
        train_all, test_size=cfg.val_split, random_state=cfg.random_seed
    )
    print(f"Train: {len(train_recs)}, Val: {len(val_recs)}, Test: {len(test_recs)}")

    DS = LSIGDatasetNoFFT if no_fft else LSIGDataset
    ds_train = DS(train_recs, cfg, augment=True)
    ds_val   = DS(val_recs,   cfg, augment=False)
    ds_test  = DS(test_recs,  cfg, augment=False)

    return ds_train, ds_val, ds_test, train_recs


# ─── Pos-weight for class imbalance ─────────────────────
def compute_pos_weight(records: List[PacketRecord]) -> torch.Tensor:
    bits = np.stack([r.lsig_bits for r in records], axis=0)
    pos  = bits.sum(axis=0)
    neg  = bits.shape[0] - pos
    pw   = neg / np.maximum(pos, 1.0)
    return torch.tensor(pw, dtype=torch.float32)
