"""Model evaluation script for DPO‑Finance.

Calculates *pairwise directional accuracy*, *pairwise logistic loss*, and *ROC‑AUC*
(on good vs. bad event scores) using the preference‑pair dataset.

Usage example::

    python src/eval.py \
        --checkpoint runs/2025-04-21/epoch=29-step=5000.ckpt \
        --pairs-file data/pairs.parquet

This minimal evaluator focuses on the ranking objective itself.  For a full
trading back‑test (Sharpe ratio, drawdown, etc.) see `experiments/backtest.ipynb`
or extend this script accordingly.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, log_loss
from tqdm import tqdm

from dpo_forecasting.models.dpo_model import DPOModel
from dpo_forecasting.data.dataset import PairwiseDataset as PreferencePairDataset
from dpo_forecasting.utils.device import get_device
# ───────────────────────────────────────────── CLI ──

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate DPO model on preference pairs.")
    p.add_argument("--checkpoint", required=True, type=str, help="Path to Lightning checkpoint (.ckpt)")
    p.add_argument("--pairs-file", required=True, type=str)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


# ─────────────────────────────────────────── eval ──

def evaluate(model: DPOModel, dl: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    acc_cnt = 0
    total = 0
    y_true = []  # 1 for good, 0 for bad
    y_score = []

    with torch.no_grad():
        for batch in tqdm(dl, desc="Eval", unit="batch"):
            # Support both dict batch (old) and tuple batch (new PairwiseDataset)
            if isinstance(batch, (list, tuple)):
                good, bad = batch[0].float().to(device, non_blocking=True), batch[1].float().to(device, non_blocking=True)
            else:
                good, bad = batch["good"].float().to(device, non_blocking=True), batch["bad"].float().to(device, non_blocking=True)
            s_good = model(good)
            s_bad = model(bad)
            acc_cnt += (s_good > s_bad).sum().item()
            total += len(good)
            y_true.extend([1] * len(good) + [0] * len(bad))
            y_score.extend(torch.cat([s_good, s_bad]).cpu().tolist())

    acc = acc_cnt / total
    auc = roc_auc_score(y_true, y_score)
    # logistic loss on single events (sigmoid cross‑entropy)
    probs = torch.sigmoid(torch.tensor(y_score)).numpy()
    ce = log_loss(y_true, probs)

    return {"pair_accuracy": acc, "roc_auc": auc, "cross_entropy": ce}


def main():
    args = parse_args()
    device = get_device()

    # Load model (without trainer for simplicity)
    model = DPOModel.load_from_checkpoint(args.checkpoint, map_location=device)
    model = model.to(device).eval()
    print(f"[INFO] model: {model.__class__.__name__} → {device}")

    dataset = PreferencePairDataset(args.pairs_file)
    dl = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda" or device.type == "mps"),
    )

    metrics = evaluate(model, dl, device)
    print("\nEvaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.4f}")


if __name__ == "__main__":
    main()
