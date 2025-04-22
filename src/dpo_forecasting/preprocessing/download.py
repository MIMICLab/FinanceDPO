"""download.py
Robust OHLCV downloader using *yfinance*.

Features
--------
1. **Retry + fallback** – up to 3 attempts; if `yf.download` fails, falls back to
   `Ticker.history()`.
2. **Data sanitisation** – auto‑adjust off; if only `Adj Close` exists, rename it
   to `Close`, remove rows where `Close` is NaN, ensure at least 10 rows.
3. **Graceful skip** – symbols with no data after all retries are reported in the
   summary but do not raise exceptions.
4. **Sample ticker list** – if `--symbols-file` is missing, a sample list of
   FAANG symbols is written so users can run a quick demo.

CLI example
~~~~~~~~~~~
```bash
python src/dpo_forecasting/data/download.py \
    --symbols-file data/sp500.txt \
    --start 2000-01-01
```
CSV files are written to `data/raw/<TICKER>.csv`.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import yfinance as yf

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return DataFrame with a single numeric 'Close' column.
    Handles yfinance quirks: MultiIndex columns, Adj Close only, etc.
    """
    if df.empty:
        return df

    # 1) flatten MultiIndex → keep only 'Close' (or 'Adj Close')
    if isinstance(df.columns, pd.MultiIndex):
        if ("Close", "") in df.columns:
            df = df.loc[:, ("Close", "")]         # Series OR 1‑col DF
        elif ("Adj Close", "") in df.columns:
            df = df.loc[:, ("Adj Close", "")]
        else:
            return pd.DataFrame()

    # after slicing, df may be Series; wrap to DataFrame
    if isinstance(df, pd.Series):
        df = df.to_frame("Close")
    elif df.shape[1] == 1:
        df.columns = ["Close"]

    # 2) rename Adj Close
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})
    if "Close" not in df.columns:
        return pd.DataFrame()

    # 3) reset index & numeric cast
    df = df.reset_index()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])
    return df


def fetch_symbol(symbol: str, start: str, end: str, retries: int = 3) -> pd.DataFrame:
    """Download *symbol*; return empty DataFrame on failure."""
    for attempt in range(1, retries + 1):
        # primary fetch
        df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
        df = _clean(df)
        if not df.empty:
            break
        # fallback fetch
        try:
            df = yf.Ticker(symbol).history(start=start, end=end)
            df = _clean(df)
            if not df.empty:
                break
        except Exception:
            pass
        # wait then retry
        if attempt < retries:
            time.sleep(1)
    return df if len(df) >= 10 else pd.DataFrame()


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def read_symbols(path: Path) -> List[str]:
    return [s.strip().upper() for s in path.read_text().splitlines() if s.strip()]


def create_sample_symbols(path: Path) -> None:
    sample = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sample) + "\n")
    print(f"[WARN] '{path}' not found – created sample list: {', '.join(sample)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download daily OHLCV price data (yfinance)")
    p.add_argument("--symbols-file", required=True, help="Text file with one ticker per line")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"), help="End date (default: today)")
    p.add_argument("--out-dir", default="data/raw", help="Destination directory for CSV files")
    return p.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_cli()

    sym_path = Path(args.symbols_file)
    if not sym_path.exists():
        create_sample_symbols(sym_path)
    symbols = read_symbols(sym_path)

    out_dir = Path(args.out_dir)
    successes, failures = [], []

    print(f"[INFO] downloading {len(symbols)} symbols → {out_dir.resolve()}")

    for sym in symbols:
        print(f"  • {sym} …", end="", flush=True)
        df = fetch_symbol(sym, args.start, args.end)
        if df.empty:
            print(" FAILED")
            failures.append(sym)
            continue
        save_csv(df, out_dir / f"{sym}.csv")
        successes.append(sym)
        print(" ok")

    # summary
    print("\n[S U M M A R Y]")
    print(f"  success : {len(successes)}")
    print(f"  failed  : {len(failures)}")
    if failures:
        print("  failed tickers:", ", ".join(sorted(failures)))


if __name__ == "__main__":
    main()
