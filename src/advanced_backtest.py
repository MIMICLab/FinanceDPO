#!/usr/bin/env python
"""advanced_backtest.py — v4 *complete*

Versatile portfolio back‑test supporting:
• Long/short softmax weighting
• Dynamic leverage (realised vol + optional VIX scaling)
• Equity stop‑loss & trailing stop
• Transaction cost + slippage
• Device auto‑select (cuda → mps → cpu)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

from dpo_forecasting.data.extractors import ReturnWindowExtractor
from dpo_forecasting.models.dpo_model import DPOModel
from dpo_forecasting.utils.device import get_device

# ───────────────────────────── helper utilities ─────────────────────────────

def annualize(vol_daily: float) -> float:
    return vol_daily * np.sqrt(252)


def calc_metrics(ret: np.ndarray) -> Dict[str, float]:
    if len(ret) == 0:
        return {}
    ann_ret = ret.mean() * 252
    ann_vol = ret.std(ddof=1) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol else 0.0
    down = ret[ret < 0]
    sortino = ann_ret / (down.std(ddof=1) * np.sqrt(252)) if len(down) else np.inf
    equity = np.cumprod(1 + ret)
    peak = np.maximum.accumulate(equity)
    mdd = ((equity - peak) / peak).min()
    calmar = -ann_ret / mdd if mdd else np.inf
    var95 = np.percentile(ret, 5)
    hit = (ret > 0).mean()
    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": mdd,
        "VaR_95": var95,
        "hit_ratio": hit,
    }


def softmax_clip(x: np.ndarray, cap: float) -> np.ndarray:
    x[np.isneginf(x)] = -1e9
    e = np.exp(x - np.nanmax(x))
    s = e.sum()
    if s == 0 or np.isclose(s, 0):
        return np.zeros_like(x)
    w = e / s
    w = np.clip(w, 0, cap)
    s2 = w.sum()
    return w / s2 if s2 else w

# ───────────────────────────── back‑test core ───────────────────────────────

def backtest(
    model: DPOModel,
    price_dfs: Dict[str, pd.DataFrame],
    lookback: int,
    rebalance: int,
    hold: int,
    fixed_leverage: float,
    target_vol: Optional[float],
    vol_window: int,
    max_leverage: float,
    vix_series: Optional[pd.Series],
    vix_scale: float,
    long_short: bool,
    max_long: float,
    max_short: float,
    cost_bps: float,
    slip_pct: float,
    eq_stop: Optional[float],
    trail_stop: Optional[float],
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray, List[pd.Timestamp]]:
    extractor = ReturnWindowExtractor(lookback)
    common_dates = sorted(set.intersection(*[set(df.Date) for df in price_dfs.values()]))
    if vix_series is not None:
        common_dates = [d for d in common_dates if d in vix_series.index]
    if len(common_dates) < lookback + hold:
        raise RuntimeError("Insufficient overlap across symbols/VIX")

    symbols = sorted(price_dfs)
    close = np.vstack([price_dfs[s].set_index("Date").loc[common_dates]["Close"].values for s in symbols])
    high = np.vstack([price_dfs[s].set_index("Date").loc[common_dates]["High"].values for s in symbols])
    low = np.vstack([price_dfs[s].set_index("Date").loc[common_dates]["Low"].values for s in symbols])

    S, T = close.shape
    cash = 1.0
    peak_eq = 1.0
    eq_curve: List[float] = []
    port_ret_hist: List[float] = []
    pos_long = np.zeros((S, hold))
    pos_short = np.zeros((S, hold))

    for t in tqdm(range(lookback, T - hold), desc="back‑test"):
        date = common_dates[t]
        # --- realise pnl for expiring slice ---
        pnl_long = pos_long[:, 0] * (close[:, t] / close[:, t - hold] - 1)
        pnl_short = -pos_short[:, 0] * (close[:, t] / close[:, t - hold] - 1)
        day_ret = pnl_long.sum() + pnl_short.sum()  # use notional sums
        cash *= 1 + day_ret
        port_ret_hist.append(day_ret)
        pos_long = np.roll(pos_long, -1, axis=1); pos_long[:, -1] = 0
        pos_short = np.roll(pos_short, -1, axis=1); pos_short[:, -1] = 0

        # --- equity risk controls ---
        peak_eq = max(peak_eq, cash)
        if trail_stop and cash < peak_eq * (1 - trail_stop):
            break
        if eq_stop and cash < 1 - eq_stop:  # fix equity stop logic
            break

        # --- dynamic leverage ---
        lev = fixed_leverage
        if target_vol and len(port_ret_hist) >= vol_window:
            realised = np.std(port_ret_hist[-vol_window:], ddof=1)
            if realised and np.isfinite(realised) and realised > 0:  # guard against NaN realised vol
                lev = min(max_leverage, (target_vol / annualize(realised)))
        if vix_series is not None:
            lev = min(max_leverage, lev * vix_scale / vix_series.loc[date])

        # --- rebalance ---
        if (t - lookback) % rebalance == 0:
            feats = np.stack([extractor(close[i], t) for i in range(S)])
            with torch.no_grad():
                scores = model(torch.from_numpy(feats).to(device)).cpu().numpy()
            if long_short:
                w_long = softmax_clip(np.where(scores > 0, scores, -np.inf), max_long)
                w_short = softmax_clip(np.where(scores < 0, -scores, -np.inf), max_short)
            else:
                # long‑only: act only on positive scores
                w_long = softmax_clip(np.where(scores > 0, scores, -np.inf), max_long)
                w_short = np.zeros_like(w_long)
            new_long = lev * w_long
            new_short = lev * w_short
            delta = np.abs(new_long - pos_long[:, -1]) + np.abs(new_short - pos_short[:, -1])
            tc = delta * cost_bps / 1e4
            slip = slip_pct / 100 * (high[:, t] - low[:, t]) / close[:, t]
            cash *= 1 - (tc + slip).mean()
            pos_long[:, -1] = new_long
            pos_short[:, -1] = new_short

        eq_curve.append(cash)

    ret = np.diff(eq_curve) / eq_curve[:-1] if len(eq_curve) > 1 else np.array([])
    return calc_metrics(ret), np.array(eq_curve), common_dates[lookback : lookback + len(eq_curve)]

# ───────────────────────────── CLI & entry ────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Advanced back‑test v4")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prices-dir", required=True)
    p.add_argument("--lookback", type=int, default=30)
    p.add_argument("--rebalance", type=int, default=5)
    p.add_argument("--hold", type=int, default=10)

    p.add_argument("--fixed-leverage", type=float, default=1.0)
    p.add_argument("--target-vol", type=float)
    p.add_argument("--vol-window", type=int, default=20)
    p.add_argument("--max-leverage", type=float, default=3.0)
    p.add_argument("--vix-file", type=str)
    p.add_argument("--vix-scale", type=float, default=20.0)

    p.add_argument("--long-short", action="store_true")
    p.add_argument("--max-long", type=float, default=0.1)
    p.add_argument("--max-short", type=float, default=0.05)

    p.add_argument("--cost-bps", type=float, default=1.0)
    p.add_argument("--slip-pct", type=float, default=0.1)
    p.add_argument("--eq-stop", type=float)
    p.add_argument("--trail-stop", type=float)
    p.add_argument("--save-equity", type=str, help="CSV path to save equity curve")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    model = DPOModel.load_from_checkpoint(args.checkpoint, map_location=device)
    model = model.to(device).eval()

    # ------------------------------------------------------------------
    # Ensure lookback length matches the model’s expected feature dim.
    # ReturnWindowExtractor outputs (lookback - 1) simple‑return features,
    # so required lookback = in_features + 1.
    try:
        in_dim = model.net[0].in_features  # assumes first layer is nn.Linear
    except Exception:
        in_dim = None
    if in_dim is not None:
        expected_lb = in_dim + 1
        if args.lookback != expected_lb:
            print(
                f"[WARN] lookback={args.lookback} produces feature length {args.lookback - 1}, "
                f"but model expects feature dim {in_dim}. Adjusting lookback to {expected_lb}."
            )
            args.lookback = expected_lb
    # ------------------------------------------------------------------

    # load price CSVs
    price_dfs: Dict[str, pd.DataFrame] = {}
    for csv in Path(args.prices_dir).glob("*.csv"):
        df = pd.read_csv(csv, parse_dates=["Date"])
        if len(df) > args.lookback + args.hold + 1:
            price_dfs[csv.stem] = df
    if len(price_dfs) < 2:
        raise RuntimeError("Need at least 2 symbols with sufficient history")

    # optional VIX
    vix_series = None
    if args.vix_file:
        vix_df = pd.read_csv(args.vix_file, parse_dates=["Date"])
        vix_series = vix_df.set_index("Date")["Close"]

    metrics, equity, dates = backtest(
        model,
        price_dfs,
        lookback=args.lookback,
        rebalance=args.rebalance,
        hold=args.hold,
        fixed_leverage=args.fixed_leverage,
        target_vol=args.target_vol,
        vol_window=args.vol_window,
        max_leverage=args.max_leverage,
        vix_series=vix_series,
        vix_scale=args.vix_scale,
        long_short=args.long_short,
        max_long=args.max_long,
        max_short=args.max_short,
        cost_bps=args.cost_bps,
        slip_pct=args.slip_pct,
        eq_stop=args.eq_stop,
        trail_stop=args.trail_stop,
        device=device,
    )

    print("\n=== ADVANCED BACKTEST SUMMARY ===")
    for k, v in metrics.items():
        if "drawdown" in k.lower():
            print(f"{k:14s}: {v:.2%}")
        else:
            print(f"{k:14s}: {v:.4f}")

    if args.save_equity:
        out = Path(args.save_equity)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"date": dates, "equity": equity}).to_csv(out, index=False)
        print(f"Equity curve saved → {out}")


if __name__ == "__main__":
    main()
