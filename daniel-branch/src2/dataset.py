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


# ─── 数据记录 ───────────────────────────────────────────
@dataclass
class PacketRecord:
    packet_id: str
    bin_path: str
    mat_path: str
    protocol: str
    lsig_bits: Optional[np.ndarray]


# ─── 文件发现 ───────────────────────────────────────────
def discover_records(root_dir: str, split: str) -> List[PacketRecord]:
    split_dir = os.path.join(root_dir, split)
    records = []

    for exp_dir in sorted(os.listdir(split_dir)):
        if exp_dir.startswith("_"):
            continue
        exp_path = os.path.join(split_dir, exp_dir)
        low_dir = os.path.join(exp_path, "Low")
        label_dir = os.path.join(exp_path, "Label")

        if not os.path.isdir(low_dir) or not os.path.isdir(label_dir):
            continue

        for bin_path in sorted(glob.glob(os.path.join(low_dir, "Packet_*.bin"))):
            idx = os.path.basename(bin_path).replace("Packet_", "").replace(".bin", "")
            mat_path = os.path.join(label_dir, f"Packet_{idx}.mat")
            if not os.path.exists(mat_path):
                continue

            try:
                mat = loadmat(mat_path)
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


# ─── 信号处理 ───────────────────────────────────────────
def load_waveform(bin_path: str) -> np.ndarray:
    return np.fromfile(bin_path, dtype=np.complex64)


def normalize(wave: np.ndarray) -> np.ndarray:
    """零均值单位标准差归一化（DeepMon Step 1）"""
    wave = wave - wave.mean()
    std = wave.std()
    if std > 1e-8:
        wave = wave / std
    return wave


def apply_phase_augment(wave: np.ndarray) -> np.ndarray:
    """随机相位偏移（DeepMon数据增强）"""
    phase = np.random.uniform(0, 2 * np.pi)
    return wave * np.exp(1j * phase)


def wave_to_2d_spectrogram(
    wave: np.ndarray,
    n_fft: int = 64,
    n_symbols: int = 12,
) -> np.ndarray:
    """
    DeepMon Step 2: per-OFDM-symbol FFT
    输入: complex waveform [T]
    输出: [2, n_symbols, n_fft]  (real/imag两个channel)
    """
    specs = []
    for i in range(n_symbols):
        start = i * n_fft
        end = start + n_fft
        if start >= len(wave):
            seg = np.zeros(n_fft, dtype=np.complex64)
        elif end > len(wave):
            seg = np.zeros(n_fft, dtype=np.complex64)
            seg[:len(wave) - start] = wave[start:]
        else:
            seg = wave[start:end]
        specs.append(np.fft.fft(seg))

    spec = np.stack(specs, axis=0)  # [n_symbols, n_fft]
    out = np.stack([spec.real, spec.imag], axis=0)  # [2, n_symbols, n_fft]
    return out.astype(np.float32)


# ─── Dataset ────────────────────────────────────────────
class LSIGDataset(Dataset):
    def __init__(
        self,
        records: List[PacketRecord],
        cfg: Config,
        augment: bool = False,
    ):
        self.records = records
        self.cfg = cfg
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        wave = load_waveform(rec.bin_path)

        # 数据增强：随机相位偏移
        if self.augment and self.cfg.use_phase_augment:
            wave = apply_phase_augment(wave)

        # 归一化
        wave = normalize(wave)

        # FFT → 2D频谱图
        x = wave_to_2d_spectrogram(wave, self.cfg.n_fft, self.cfg.n_symbols)

        y = rec.lsig_bits.astype(np.float32)

        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
            "packet_id": rec.packet_id,
        }


# ─── 数据集构建 ──────────────────────────────────────────
def build_datasets(cfg: Config) -> Tuple:
    print("Discovering records...")
    train_all = discover_records(cfg.root_dir, "Train")
    test_recs = discover_records(cfg.root_dir, "Test")

    # 过滤Non-HT，必须有24位LSIG
    def filter_records(recs):
        return [
            r for r in recs
            if r.protocol == cfg.target_protocol
            and r.lsig_bits is not None
            and len(r.lsig_bits) == 24
        ]

    train_all = filter_records(train_all)
    test_recs = filter_records(test_recs)

    print(f"Train (before split): {len(train_all)}")
    print(f"Test:                 {len(test_recs)}")

    # 切分train/val
    train_recs, val_recs = train_test_split(
        train_all,
        test_size=cfg.val_split,
        random_state=cfg.random_seed,
    )

    print(f"Train: {len(train_recs)}, Val: {len(val_recs)}, Test: {len(test_recs)}")

    ds_train = LSIGDataset(train_recs, cfg, augment=True)
    ds_val = LSIGDataset(val_recs, cfg, augment=False)
    ds_test = LSIGDataset(test_recs, cfg, augment=False)

    return ds_train, ds_val, ds_test, train_recs


# ─── 计算pos_weight ──────────────────────────────────────
def compute_pos_weight(records: List[PacketRecord]) -> torch.Tensor:
    bits = np.stack([r.lsig_bits for r in records], axis=0)
    pos = bits.sum(axis=0)
    neg = bits.shape[0] - pos
    pw = neg / np.maximum(pos, 1.0)
    return torch.tensor(pw, dtype=torch.float32)


if __name__ == "__main__":
    cfg = Config()
    ds_train, ds_val, ds_test, train_recs = build_datasets(cfg)
    sample = ds_train[0]
    print(f"\nSample x shape: {sample['x'].shape}")  # 应该是 [2, 12, 64]
    print(f"Sample y shape: {sample['y'].shape}")    # 应该是 [24]
    print(f"First 8 bits:   {sample['y'][:8].tolist()}")