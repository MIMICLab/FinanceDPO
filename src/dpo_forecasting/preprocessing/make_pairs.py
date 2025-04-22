"""Generate preference pairs (good vs. bad outcomes) from raw price data.

The script takes daily OHLCV CSV files (as produced by ``download.py``),
computes event‑based outcomes (e.g. *K*‑day forward returns), labels each
outcome as **good** / **bad** using percentile thresholds, and finally writes a
single Parquet file containing all pair indices ready for DPO training.

Example::

    python src/dpo_forecasting/data/make_pairs.py \
        --prices-dir data/raw \
        --lookahead 20 --lookback 31 \
        --good-quantile 0.8 --bad-quantile 0.2 \
        --out-file data/pairs_20d.parquet

The resulting Parquet has four columns:

``symbol``  | ``idx_good`` | ``idx_bad`` | ``label``
----------- | ------------ | ---------- | ----------
AAPL        | 10345        | 9284       | 1
…           | …            | …          | …

where *idx* are integer locations into the per‑symbol DataFrame (these are more
compact than full contexts; the actual features will be gathered lazily during
training).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from extractors import ReturnWindowExtractor
import torch


# ───────────────────────────────────────────────────────────── CLI parsing ──

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create good/bad preference pairs from OHLCV CSVs.")
    p.add_argument("--prices-dir", required=True, type=str, help="Directory of CSV files from download.py")
    p.add_argument("--lookahead", type=int, default=20, help="Forward horizon in trading days (default: 20)")
    p.add_argument("--lookback", type=int, default=31,
                   help="Number of close prices per window (default: 31 → feature dim 30)")
    p.add_argument("--good-quantile", type=float, default=0.8, help="Quantile for good label (default: 0.8)")
    p.add_argument("--bad-quantile", type=float, default=0.2, help="Quantile for bad label (default: 0.2)")
    p.add_argument("--min-samples", type=int, default=50, help="Skip symbols with fewer samples than this")
    p.add_argument("--out-file", type=str, default="data/pairs.parquet", help="Output Parquet file")
    p.add_argument("--cache-file", type=str, default="data/pairs_cache.pt",
                   help="Torch cache file for fast loading (default: data/pairs_cache.pt)")
    p.add_argument("--skip-parquet", action="store_true",
                   help="If set, do not write the intermediate Parquet file")
    return p.parse_args()


# ───────────────────────────────────────────────────────── helpers ──

def load_price_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])  # columns: Date, Open, High, Low, Close, Adj Close, Volume
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["Close"] = (
        df["Close"]
        .astype(str)        
        .str.replace(",", "") # remove commas
        .str.strip()
        .replace({"": np.nan, "-": np.nan})
        .astype(float)    
    )
    df.dropna(subset=["Close"], inplace=True)   
    return df

def compute_forward_return(df: pd.DataFrame, lookahead: int) -> pd.Series:
    close = df["Close"].values
    fwd = (np.roll(close, -lookahead) - close) / close
    fwd[-lookahead:] = np.nan  # last lookahead rows have undefined forward return
    return pd.Series(fwd, name=f"ret_{lookahead}d")


def label_good_bad(ret: pd.Series, good_q: float, bad_q: float) -> Tuple[np.ndarray, np.ndarray]:
    good_thresh = ret.quantile(good_q)
    bad_thresh = ret.quantile(bad_q)
    good_idx = np.where(ret >= good_thresh)[0]
    bad_idx = np.where(ret <= bad_thresh)[0]
    return good_idx, bad_idx


def make_pairs_for_symbol(df: pd.DataFrame, lookahead: int, gq: float, bq: float, lookback: int) -> pd.DataFrame:
    ret = compute_forward_return(df, lookahead)
    df = df.assign(fwd_ret=ret)
    good_idx, bad_idx = label_good_bad(df["fwd_ret"], gq, bq)

    # drop indices that don't have enough history for the look‑back window
    good_idx = good_idx[good_idx >= lookback]
    bad_idx  = bad_idx[bad_idx  >= lookback]

    if len(good_idx) == 0 or len(bad_idx) == 0:
        return pd.DataFrame(columns=["idx_good", "idx_bad"])

    # Cartesian product → sample equal number of pairs for efficiency
    n_pairs = min(len(good_idx), len(bad_idx))
    rng = np.random.default_rng(42)
    good_sample = rng.choice(good_idx, size=n_pairs, replace=False)
    bad_sample = rng.choice(bad_idx, size=n_pairs, replace=False)

    pairs = pd.DataFrame({"idx_good": good_sample, "idx_bad": bad_sample})
    pairs["label"] = 1  # DPO convention: 1 means idx_good is preferred

    # pre‑compute feature vectors
    extractor = ReturnWindowExtractor(lookback)
    close = df["Close"].values.astype("float32")
    feat_g_list: List[np.ndarray] = []
    feat_b_list: List[np.ndarray] = []

    for g, b in zip(pairs["idx_good"], pairs["idx_bad"]):
        feat_g_list.append(extractor(close, int(g)))
        feat_b_list.append(extractor(close, int(b)))

    pairs["feat_good"] = feat_g_list
    pairs["feat_bad"]  = feat_b_list

    return pairs


# ───────────────────────────────────────────────────────────── main ──

def main() -> None:
    args = parse_args()
    print(f"[INFO] lookahead={args.lookahead}, lookback={args.lookback}, "
          f"pairs→{Path(args.out_file).name}, cache→{Path(args.cache_file).name}")
    prices_dir = Path(args.prices_dir)
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    all_pairs: List[pd.DataFrame] = []

    for csv_path in prices_dir.glob("*.csv"):
        symbol = csv_path.stem.upper()
        df = load_price_csv(csv_path)
        if len(df) < args.min_samples:
            continue
        pairs = make_pairs_for_symbol(
            df, args.lookahead, args.good_quantile, args.bad_quantile, args.lookback
        )
        if pairs.empty:
            continue
        pairs.insert(0, "symbol", symbol)
        all_pairs.append(pairs)
        print(f"[OK] {symbol}: {len(pairs)} pairs")

    if not all_pairs:
        raise RuntimeError("No pairs generated – check thresholds and data")

    result = pd.concat(all_pairs, ignore_index=True)

    # ----------------------------------- save parquet (optional)
    if not args.skip_parquet:
        result.to_parquet(out_file, index=False)
        print(f"[DONE] Wrote {len(result):,} pairs → {out_file}")

    # ----------------------------------- save torch cache
    fg = torch.tensor(np.asarray(result["feat_good"].to_list(), dtype=np.float32))
    fb = torch.tensor(np.asarray(result["feat_bad"].to_list(),  dtype=np.float32))
    cache_obj = {
        "feat_good": fg,
        "feat_bad":  fb,
        "good_idx":  torch.arange(len(fg)),
        "bad_idx":   torch.arange(len(fb)),
    }
    torch.save(cache_obj, args.cache_file)
    print(f"[DONE] Cached tensors → {args.cache_file}")


if __name__ == "__main__":
    main()
