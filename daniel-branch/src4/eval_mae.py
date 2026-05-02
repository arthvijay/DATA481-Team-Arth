"""
eval_mae.py  —  Evaluate trained RIS-MAE checkpoints on the test set.
Checkpoints are in ~/outputs_mae/ (written by train_mae.py run from ~/).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from dataset import build_datasets
from model_mae import IQMaskedAutoencoder
from lsig_decode import compute_tx_time_mae, compute_length_mae

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
MAE_DIR  = os.path.join(os.path.expanduser("~"), "outputs_mae")


def evaluate(model, dl, tag):
    model.eval()
    all_preds, all_gts = [], []
    with torch.no_grad():
        for batch in dl:
            x = batch["x"].to(DEVICE)
            y = batch["y"].to(DEVICE)
            all_preds.append(torch.sigmoid(model.forward_finetune(x)).cpu())
            all_gts.append(y.cpu())
    all_preds = torch.cat(all_preds)
    all_gts   = torch.cat(all_gts)

    hard   = (all_preds >= 0.5).float()
    ba     = (hard == all_gts).float().mean().item()
    ex     = (hard == all_gts).all(dim=1).float().mean().item()
    tx_mae = compute_tx_time_mae(all_preds.numpy(), all_gts.numpy())
    l_mae  = compute_length_mae(all_preds.numpy(),  all_gts.numpy())

    print(f"  [{tag:8s}]  BitAcc={ba:.4f}  ExactMatch={ex:.4f}  "
          f"TxMAE={tx_mae['mae_ms']:.4f}ms  LenMAE={l_mae['mae_bytes']:.1f}B")
    return ba, ex, tx_mae["mae_ms"], l_mae["mae_bytes"]


def main():
    cfg = Config()
    print(f"Device: {DEVICE}")
    print(f"Checkpoint dir: {MAE_DIR}")

    _, _, ds_test, _ = build_datasets(cfg, no_fft=True)
    dl_test = DataLoader(ds_test, cfg.batch_size, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=True)
    print(f"Test samples: {len(ds_test)}")

    model = IQMaskedAutoencoder(
        seq_len=768, patch_size=16, embed_dim=128,
        encoder_depth=4, decoder_depth=2, num_heads=4,
        mask_ratio=0.5, num_bits=24,
    ).to(DEVICE)

    results = {}
    print(f"\n{'='*60}")
    print("RIS-MAE  —  Test Set Evaluation")
    print(f"{'='*60}")

    for tag, fname in [("frozen", "best_frozen.pt"), ("full", "best_full.pt")]:
        ckpt = os.path.join(MAE_DIR, fname)
        if not os.path.exists(ckpt):
            print(f"  [{tag}]  checkpoint not found: {ckpt}")
            continue
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        results[tag] = evaluate(model, dl_test, tag)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<20} {'BitAcc':>8} {'ExMatch':>9} {'TxMAE(ms)':>11} {'LenMAE(B)':>11}")
    print(f"  {'-'*60}")
    for tag, (ba, ex, tx, lm) in results.items():
        print(f"  RIS-MAE [{tag:<6}]   {ba:>8.4f} {ex:>9.4f} {tx:>11.4f} {lm:>11.1f}")
    print(f"  {'DeepMon baseline':<20} {'0.9467':>8} {'0.657':>9} {'0.077':>11} {'~':>11}")

    # Write results to a file for final_comparison.py to read
    out_path = os.path.join(MAE_DIR, "eval_results.txt")
    with open(out_path, "w") as f:
        for tag, (ba, ex, tx, lm) in results.items():
            f.write(f"{tag} {ba:.6f} {ex:.6f} {tx:.6f} {lm:.2f}\n")
    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
