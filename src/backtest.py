"""Simple daily back‑test for a preference‑score model.

Strategy logic – per symbol, per day::
    score = model(feature_window)
    if score > upper_threshold  →  long 1 unit (notional)
    elif score < lower_threshold →  short 1 unit
    else                         →  flat

Positions are held **one trading day**; P&L is joined across all symbols with
equal notional weight.  Transaction cost is a fixed *bps* per round‑trip.

Usage example::

    python src/backtest.py \
        --checkpoint runs/best.ckpt \
        --prices-dir data/raw \
        --lookback 30 \
        --upper 0.7 --lower 0.3
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dpo_forecasting.models.dpo_model import DPOModel
from dpo_forecasting.data.dataset import ReturnWindowExtractor


# ─────────────────────────────────────────────── metrics ──

def sharpe(returns: np.ndarray, risk_free: float = 0.0) -> float:
    if returns.std(ddof=1) == 0:
        return 0.0
    daily_excess = returns - risk_free / 252
    ann_ret = daily_excess.mean() * 252
    ann_vol = returns.std(ddof=1) * np.sqrt(252)
    return ann_ret / ann_vol


def sortino(returns: np.ndarray) -> float:
    neg = returns[returns < 0]
    if len(neg) == 0:
        return np.inf
    ann_ret = returns.mean() * 252
    dd = neg.std(ddof=1) * np.sqrt(252)
    return ann_ret / dd


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    return drawdown.min()


# ─────────────────────────────────────────────── helper ──

def load_close(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    return df["Close"].values.astype(np.float32)


# ───────────────────────────────────────────── backtest ──

def backtest(
    model: DPOModel,
    prices_dir: Path,
    lookback: int,
    upper_p: float,
    lower_p: float,
    cost_bps: float = 0.0,
):
    model.eval()
    device = next(model.parameters()).device
    extractor = ReturnWindowExtractor(lookback)

    all_daily_ret: List[np.ndarray] = []  # list over symbols

    for csv_path in tqdm(list(prices_dir.glob("*.csv")), desc="Symbols"):
        close = load_close(csv_path)
        if len(close) <= lookback + 1:
            continue
        # compute features and scores
        feats = []
        for idx in range(lookback, len(close) - 1):
            feat = extractor(close, idx)
            feats.append(feat)
        X = torch.from_numpy(np.stack(feats)).to(device)
        with torch.no_grad():
            scores = model(X).cpu().numpy()
        # convert percentiles to thresholds per symbol distribution
        lo_th, hi_th = np.quantile(scores, lower_p), np.quantile(scores, upper_p)
        pos = np.where(scores > hi_th, 1, np.where(scores < lo_th, -1, 0))
        # daily returns next day
        pct_change = (close[lookback + 1 :] - close[lookback:-1]) / close[lookback:-1]
        # strategy returns net of transaction cost
        ret = pos * pct_change - np.abs(np.diff(np.concatenate([[0], pos]))) * cost_bps / 1e4
        all_daily_ret.append(ret)

    # align lengths by padding with zeros at left
    max_len = max(map(len, all_daily_ret))
    aligned = np.zeros((len(all_daily_ret), max_len), dtype=np.float32)
    for i, r in enumerate(all_daily_ret):
        aligned[i, -len(r) :] = r
    portfolio_ret = aligned.mean(axis=0)
    equity = np.cumprod(1 + portfolio_ret)

    metrics = {
        "ann_return": (portfolio_ret.mean() * 252),
        "ann_vol": portfolio_ret.std(ddof=1) * np.sqrt(252),
        "sharpe": sharpe(portfolio_ret),
        "sortino": sortino(portfolio_ret),
        "max_drawdown": max_drawdown(equity),
    }
    return metrics, equity


# ─────────────────────────────────────────────── CLI ──

def parse_args():
    p = argparse.ArgumentParser(description="Daily long/short back‑test of DPO model")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prices-dir", required=True)
    p.add_argument("--lookback", type=int, default=30)
    p.add_argument("--upper", type=float, default=0.7, help="Upper score percentile for long entry")
    p.add_argument("--lower", type=float, default=0.3, help="Lower score percentile for short entry")
    p.add_argument("--cost-bps", type=float, default=2.0, help="Round‑trip transaction cost in bps")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DPOModel.load_from_checkpoint(args.checkpoint, map_location=device)
    metrics, equity = backtest(
        model,
        Path(args.prices_dir),
        lookback=args.lookback,
        upper_p=args.upper,
        lower_p=args.lower,
        cost_bps=args.cost_bps,
    )

    print("\nBack‑test summary:")
    for k, v in metrics.items():
        if k == "max_drawdown":
            print(f"  {k:12s}: {v:.2%}")
        else:
            print(f"  {k:12s}: {v:.4f}")


if __name__ == "__main__":
    main()
