"""Advanced portfolio back‑test engine for DPO‑Finance (v4).

### New Extensions
1. **VIX‑Aware Dynamic Leverage**  (`--vix-file`, `--vix-scale`)
   * Leverage is additionally scaled by `vix_scale / VIX_t` (capped by `max_leverage`).
   * Lower VIX → higher allowable leverage; high VIX → risk reduction.
2. **Equity‑level Stop‑loss / Trailing‑stop** (`--eq-stop`, `--trail-stop`)
   * `eq-stop` : if cumulative drawdown falls below this threshold (e.g. `-0.2`), all positions are cleared and simulation terminates.
   * `trail-stop` : if equity falls more than X percent below its rolling peak, positions are liquidated and leverage set to zero for the rest of back‑test.

```
python src/advanced_backtest.py \
  --checkpoint runs/best.ckpt --prices-dir data/raw \
  --long-short --target-vol 12 --vix-file data/VIX.csv --vix-scale 20 \
  --eq-stop -0.25 --trail-stop 0.15
```
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dpo_forecasting.models.dpo_model import DPOModel
from dpo_forecasting.data.dataset import ReturnWindowExtractor

# ╭──────────────────────── helper functions ─────────────────────────╮

def annualize(vol_daily: float) -> float:
    return vol_daily * np.sqrt(252)


def calc_metrics(ret: np.ndarray) -> Dict[str, float]:
    ann_ret = ret.mean() * 252
    ann_vol = ret.std(ddof=1) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol else 0.0
    sortino = ann_ret / (ret[ret < 0].std(ddof=1) * np.sqrt(252) or np.inf)
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
    if np.all(np.isneginf(x)):
        return np.zeros_like(x)
    exp = np.exp(x - np.nanmax(x))
    w = exp / exp.sum() if exp.sum() else exp
    w = np.clip(w, 0, cap)
    if w.sum() == 0:
        return w
    return w / w.sum()

# ╰────────────────────────────────────────────────────────────────────╯

# ╭──────────────────────── back‑test core (v4) ───────────────────────╮

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
) -> Tuple[Dict[str, float], np.ndarray, List[pd.Timestamp]]:
    device = next(model.parameters()).device
    extractor = ReturnWindowExtractor(lookback)

    common_dates = sorted(set.intersection(*[set(df.Date) for df in price_dfs.values()]))
    if vix_series is not None:
        common_dates = [d for d in common_dates if d in vix_series.index]
    symbols = sorted(price_dfs)
    S, T = len(symbols), len(common_dates)

    close = np.vstack([price_dfs[s].set_index("Date").loc[common_dates]["Close"].values for s in symbols])
    high = np.vstack([price_dfs[s].set_index("Date").loc[common_dates]["High"].values for s in symbols])
    low = np.vstack([price_dfs[s].set_index("Date").loc[common_dates]["Low"].values for s in symbols])

    cash = 1.0
    equity_curve = []
    pos_long = np.zeros((S, hold))
    pos_short = np.zeros((S, hold))
    port_ret_hist: List[float] = []
    peak_equity = 1.0

    for t in tqdm(range(lookback, T - hold), desc="Days"):
        date = common_dates[t]

        # 1. Realize P&L
        pnl_long = pos_long[:, 0] * (close[:, t] / close[:, t - hold] - 1)
        pnl_short = -pos_short[:, 0] * (close[:, t] / close[:, t - hold] - 1)
        day_ret = (pnl_long.mean() + pnl_short.mean())
        cash *= 1 + day_ret
        port_ret_hist.append(day_ret)
        pos_long = np.roll(pos_long, -1, axis=1); pos_long[:, -1] = 0
        pos_short = np.roll(pos_short, -1, axis=1); pos_short[:, -1] = 0

        # Trailing‑stop check
        peak_equity = max(peak_equity, cash)
        if trail_stop and (cash < peak_equity * (1 - trail_stop)):
            print(f"Trailing‑stop hit on {date.date()} – liquidation.")
            break
        # Equity stop‑loss check
        if eq_stop and (cash < 1 + eq_stop):
            print(f"Equity stop‑loss hit ({eq_stop:.0%}) on {date.date()} – liquidation.")
            break

        # 2. Compute dynamic leverage
        if target_vol and len(port_ret_hist) >= vol_window:
            realized = np.std(port_ret_hist[-vol_window:], ddof=1)
            lev = target_vol / annualize(realized) if realized else max_leverage
        else:
            lev = fixed_leverage
        # VIX scaling
        if vix_series is not None:
            vix_today = vix_series.loc[date]
            lev *= vix_scale / vix_today
        lev = np.clip(lev, 0, max_leverage)

        # 3. Rebalance
        if (t - lookback) % rebalance == 0:
            feats = np.stack([extractor(close[i], t) for i in range(S)])
            with torch.no_grad():
                scores = model(torch.from_numpy(feats).to(device)).cpu().numpy()
            if long_short:
                w_long = softmax_clip(np.where(scores > 0, scores, -np.inf), max_long)
                w_short = softmax_clip(np.where(scores < 0, -scores, -np.inf), max_short)
            else:
                w_long = softmax_clip(scores, max_long)
                w_short = np.zeros_like(w_long)
            new_long = lev * w_long
            new_short = lev * w_short
            # 비용 적용
            delta = np.abs(new_long - pos_long[:, -1]) + np.abs(new_short - pos_short[:, -1])
            tc = delta * cost_bps / 1e4
            slip = slip_pct / 100 * (high[:, t] - low[:, t]) / close[:, t]
            cash *= 1 - (tc + slip).mean()
            pos_long[:, -1] = new_long
            pos_short[:, -1] = new_short

        equity_curve.append(cash)

    returns = np.diff(equity_curve) / equity_curve[:-1] if len(equity_curve) > 1 else np.array([])
    metrics = calc_metrics(returns) if len(returns) else {}
    return metrics, np.array(equity_curve), common_dates[lookback : lookback + len(equity_curve)]

# ╰────────────────────────────────────────────────────────────────────╯

# ╭──────────────────────────── CLI parser ────────────────────────────╮

def parse_args():
    p = argparse.ArgumentParser(description="Advanced back‑test v4 (VIX leverage, stops)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prices-dir", required=True)
    p.add_argument("--lookback", type=int, default=30)
    p.add_argument("--rebalance", type=int, default=5)
    p.add_argument("--hold", type=int, default=10)
    # leverage settings
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--target-vol", type=float)
    p.add_argument("--vol-window", type=int, default=20)
    p.add_argument("--max-leverage", type=float, default=3.0)
    p.add_argument("--vix-file", type=str)
    p.add_argument("--vix-scale", type=float, default=20.0)
    # positioning
    p.add_argument("--long-short", action="store_true")
    p.add_argument("--max-long-weight", type=float, default=0.1)
    p.add_argument("--max-short-weight", type=float, default=0.05)
    # risk controls
    p.add_argument("--eq-stop", type=float, help="Stop if equity drawdown below (negative) fraction, e.g. -0.25")
    p.add_argument("--trail-stop", type=float, help="Trailing stop percent (0.15 = 15%)")
    # costs
    p.add_argument("--cost-bps", type=float, default=1.0)
    p.add_argument("--slip-pct", type=float, default=0.0)
    return p.parse_args()

# ╰────────────────────────────────────────────────────────────────────╯

if __name__ == "__main__":
    a = parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = DPOModel.load_from_checkpoint(a.checkpoint, map_location=dev)

    price_dfs = {csv.stem: pd.read_csv(csv, parse_dates=["Date"]) for csv in Path(a.prices_dir).glob("*.csv")}
    price_dfs = {k: v for k, v in price_dfs.items() if len(v) > a.lookback + a.hold + 1}
    if len(price_dfs) < 2:
        raise RuntimeError("Insufficient symbol data")

    vix_series = None
    if a.vix_file:
        vix_df = pd.read_csv(a.vix_file, parse_dates=["Date"])
        vix_series = vix_df.set_index("Date")["Close"]

    metrics, equity, dates = backtest(
        model,
        price_dfs,
        lookback=a.lookback,
        rebalance=a.rebalance,
        hold=a.hold,
        fixed_leverage=a.leverage,
        target_vol=a.target_vol,
        vol_window=a.vol_window,
        max_le
