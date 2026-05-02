"""
lsig_decode.py
==============
Post-processing: convert predicted/ground-truth L-SIG bits
→ human-readable packet properties (rate, length, Tx time).

L-SIG bit layout (802.11-2016, Section 17.3.4, LSB-first):
  bits[ 0: 4]  Rate     (4 bits, LSB first)
  bits[ 4]     Reserved (always 0)
  bits[ 5:17]  Length   (12 bits, LSB first, PSDU length in bytes)
  bits[17]     Parity   (even parity over bits 0-16)
  bits[18:24]  Tail     (always 000000)

Tx time formula (Non-HT / legacy 802.11a):
  T_PREAMBLE = 16 μs  (L-STF 8μs + L-LTF 8μs)
  T_SIGNAL   =  4 μs  (L-SIG symbol)
  T_SYM      =  4 μs  (each data OFDM symbol)
  N_DBPS     = data bits per OFDM symbol (depends on rate)
  N_SYM      = ceil((16 + 8*LENGTH + 6) / N_DBPS)
  TxTime     = T_PREAMBLE + T_SIGNAL + T_SYM * N_SYM
             = 20 + 4 * N_SYM  (μs)
"""

import math
import numpy as np
import torch
from typing import Union

# ─── Rate lookup (bits 0-3, LSB first → Mbps) ───────────
# Key: tuple of 4 bits (b0, b1, b2, b3)
RATE_MAP = {
    (1, 1, 0, 1): 6,
    (1, 1, 1, 1): 9,
    (0, 1, 0, 1): 12,
    (0, 1, 1, 1): 18,
    (1, 0, 0, 1): 24,
    (1, 0, 1, 1): 36,
    (0, 0, 0, 1): 48,
    (0, 0, 1, 1): 54,
}

# Data bits per OFDM symbol (N_DBPS) per rate
N_DBPS_MAP = {
    6:  24,
    9:  36,
    12: 48,
    18: 72,
    24: 96,
    36: 144,
    48: 192,
    54: 216,
}


def bits_to_int(bits: np.ndarray) -> int:
    """Convert LSB-first bit array to integer."""
    return int(sum(int(b) << i for i, b in enumerate(bits)))


def decode_lsig(bits: np.ndarray) -> dict:
    """
    Decode one 24-bit L-SIG array (float32 or int, thresholded at 0.5).

    Args:
        bits: array of shape [24], values in {0, 1} or probabilities

    Returns dict with keys:
        rate_mbps   : float or None if unknown rate code
        length_bytes: int
        tx_time_us  : float or None
        parity_ok   : bool
        rate_bits   : list[int]
        length_bits : list[int]
        parity_bit  : int
        tail_bits   : list[int]
    """
    bits = np.array(bits).flatten()
    assert len(bits) == 24, f"Expected 24 bits, got {len(bits)}"
    bits = (bits >= 0.5).astype(int)   # threshold probabilities

    rate_bits   = bits[0:4].tolist()
    length_bits = bits[5:17]
    parity_bit  = int(bits[17])
    tail_bits   = bits[18:24].tolist()

    # Rate
    rate_key  = tuple(rate_bits)
    rate_mbps = RATE_MAP.get(rate_key, None)

    # Length (12 bits, LSB first)
    length_bytes = bits_to_int(length_bits)

    # Parity check (even parity over bits 0-16)
    parity_ok = (int(bits[:17].sum()) % 2) == 0

    # Tx time
    tx_time_us = None
    if rate_mbps is not None:
        n_dbps    = N_DBPS_MAP[rate_mbps]
        n_sym     = math.ceil((16 + 8 * length_bytes + 6) / n_dbps)
        tx_time_us = 20.0 + 4.0 * n_sym   # μs

    return {
        "rate_mbps":    rate_mbps,
        "length_bytes": length_bytes,
        "tx_time_us":   tx_time_us,
        "parity_ok":    parity_ok,
        "rate_bits":    rate_bits,
        "length_bits":  length_bits.tolist(),
        "parity_bit":   parity_bit,
        "tail_bits":    tail_bits,
    }


def batch_decode_lsig(bits_batch: Union[np.ndarray, torch.Tensor]) -> list:
    """
    Decode a batch of L-SIG predictions.

    Args:
        bits_batch: [B, 24] array/tensor (probabilities or hard bits)

    Returns:
        list of B dicts (same format as decode_lsig)
    """
    if isinstance(bits_batch, torch.Tensor):
        bits_batch = bits_batch.detach().cpu().numpy()
    return [decode_lsig(bits_batch[i]) for i in range(len(bits_batch))]


def compute_tx_time_mae(
    pred_bits: Union[np.ndarray, torch.Tensor],
    true_bits: Union[np.ndarray, torch.Tensor],
) -> dict:
    """
    Compute Tx time MAE between predicted and ground-truth L-SIG bits.
    Skips samples where rate is unrecognised in either pred or GT.

    Args:
        pred_bits: [B, 24] predicted probabilities or hard bits
        true_bits: [B, 24] ground-truth bits

    Returns dict:
        mae_us        : mean absolute error in μs
        mae_ms        : mean absolute error in ms
        n_valid       : number of samples with decodable rate
        n_skipped     : samples skipped (unknown rate code)
        errors_us     : array of per-sample absolute errors
    """
    pred_decoded = batch_decode_lsig(pred_bits)
    true_decoded = batch_decode_lsig(true_bits)

    errors = []
    n_skipped = 0

    for p, t in zip(pred_decoded, true_decoded):
        if p["tx_time_us"] is None or t["tx_time_us"] is None:
            n_skipped += 1
            continue
        errors.append(abs(p["tx_time_us"] - t["tx_time_us"]))

    errors = np.array(errors)
    mae_us = float(errors.mean()) if len(errors) > 0 else float("nan")

    return {
        "mae_us":    mae_us,
        "mae_ms":    mae_us / 1000.0,
        "n_valid":   len(errors),
        "n_skipped": n_skipped,
        "errors_us": errors,
    }


def compute_length_mae(
    pred_bits: Union[np.ndarray, torch.Tensor],
    true_bits: Union[np.ndarray, torch.Tensor],
) -> dict:
    """
    Compute PSDU length MAE in bytes between predicted and ground-truth bits.

    Returns dict:
        mae_bytes : mean absolute error in bytes
        mae_kb    : mean absolute error in kilobytes
        n_valid   : number of samples
        errors    : array of per-sample absolute errors
    """
    pred_decoded = batch_decode_lsig(pred_bits)
    true_decoded = batch_decode_lsig(true_bits)

    errors = np.array([
        abs(p["length_bytes"] - t["length_bytes"])
        for p, t in zip(pred_decoded, true_decoded)
    ])

    return {
        "mae_bytes": float(errors.mean()),
        "mae_kb":    float(errors.mean()) / 1024.0,
        "n_valid":   len(errors),
        "errors":    errors,
    }


# ─── Quick test ─────────────────────────────────────────
if __name__ == "__main__":
    # Ground truth from Packet_1_Label.mat
    gt_bits = np.array([1,1,0,1,0,0,0,1,1,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0], dtype=np.float32)
    result  = decode_lsig(gt_bits)

    print("=== Packet_1 L-SIG decode ===")
    print(f"  Rate     : {result['rate_mbps']} Mbps  (bits: {result['rate_bits']})")
    print(f"  Length   : {result['length_bytes']} bytes")
    print(f"  Parity   : {'OK' if result['parity_ok'] else 'FAIL'}  (bit={result['parity_bit']})")
    print(f"  Tx time  : {result['tx_time_us']:.1f} μs  =  {result['tx_time_us']/1000:.4f} ms")
    print(f"  Tail bits: {result['tail_bits']}  (should be all zeros)")

    # Simulate a slightly wrong prediction (flip 2 bits)
    pred_bits = gt_bits.copy()
    pred_bits[6] = 1 - pred_bits[6]   # flip a length bit
    pred_bits[8] = 1 - pred_bits[8]

    result2 = decode_lsig(pred_bits)
    print("\n=== Simulated prediction (2 bits flipped) ===")
    print(f"  Rate     : {result2['rate_mbps']} Mbps")
    print(f"  Length   : {result2['length_bytes']} bytes  (GT: {result['length_bytes']})")
    print(f"  Tx time  : {result2['tx_time_us']:.1f} μs  (GT: {result['tx_time_us']:.1f} μs)")
    print(f"  Tx error : {abs(result2['tx_time_us'] - result['tx_time_us']):.1f} μs")

    # Batch MAE test
    B = 8
    preds = np.tile(pred_bits, (B, 1))
    gts   = np.tile(gt_bits,   (B, 1))
    tx_mae = compute_tx_time_mae(preds, gts)
    l_mae  = compute_length_mae(preds, gts)
    print(f"\n=== Batch MAE (B={B}) ===")
    print(f"  Tx Time MAE : {tx_mae['mae_us']:.2f} μs  ({tx_mae['mae_ms']:.4f} ms)")
    print(f"  Length  MAE : {l_mae['mae_bytes']:.1f} bytes  ({l_mae['mae_kb']:.4f} kB)")
