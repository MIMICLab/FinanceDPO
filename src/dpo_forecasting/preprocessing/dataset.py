"""Preference Dataset / DataModule — robust."""

from __future__ import annotations
from pathlib import Path
from typing import List
from dpo_forecasting.preprocessing.extractors import ReturnWindowExtractor

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl


def _sanitize_features(df: pd.DataFrame, label_col: str, verbose: bool = False) -> torch.Tensor:
    """Return float32 tensor of numeric features; drop non‑numeric."""
    X = df.drop(label_col, axis=1)
    keep: List[str] = []
    for col in X.columns:
        if np.issubdtype(X[col].dtype, np.number):
            keep.append(col)
        else:
            # try coercion
            numeric = pd.to_numeric(X[col], errors="coerce")
            if numeric.notna().any():
                X[col] = numeric
                keep.append(col)
            else:
                if verbose:
                    print(f"[WARN] drop non‑numeric column: {col}")
    if not keep:
        raise ValueError("No numeric feature columns after cleaning.")
    return torch.tensor(X[keep].values, dtype=torch.float32)


class PreferenceDataset(Dataset):
    def __init__(self, parquet_file: str, prices_dir: str, lookback: int = 31):
        df = pd.read_parquet(parquet_file)
        
        if "label" not in df.columns:
            raise ValueError("pairs parquet must contain a `label` column")
        label_col = "label"

        extractor = ReturnWindowExtractor(lookback)
        features: List[np.ndarray] = []
        labels:   List[int] = []
        drop_rows: List[int] = []

        for idx, row in df.iterrows():
            sym = row["symbol"]
            ix  = int(row["idx_good"] if "idx_good" in row else row.name)
            csv_path = Path(prices_dir) / f"{sym}.csv"
            if not csv_path.exists():
                drop_rows.append(idx)
                continue
            close = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=4)
            if ix < lookback:
                drop_rows.append(idx)
                continue
            feat = extractor(close, ix)
            features.append(feat)
            labels.append(int(row[label_col]))

        if drop_rows:
            df = df.drop(index=drop_rows).reset_index(drop=True)

        self.X = torch.tensor(np.stack(features), dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)
        self.num_features = self.X.shape[1]
        self.df = df
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── src/dpo_forecasting/data/dataset.py ────────────────────────────────
class PairwiseDataset(Dataset):
    """
    If the DataFrame has explicit `idx_good` / `idx_bad` columns, use them
    directly. Otherwise fall back to the old behaviour that relies on a
    binary `label` column.
    """
    def __init__(self, pairs_src: pd.DataFrame):
        # Accept: (a) PreferenceDataset, (b) raw DataFrame, or (c) a path‑like to a parquet file
        if isinstance(pairs_src, PreferenceDataset):
            self._pref_ds = pairs_src
            pairs_df = pairs_src.df
        else:
            self._pref_ds = None
            if isinstance(pairs_src, (str, Path)):
                # Read only the columns we need to minimise I/O
                base_cols = ["symbol", "label", "idx_good", "idx_bad",
                             "feat_good", "feat_bad"]
                try:
                    pairs_df = pd.read_parquet(
                        pairs_src, columns=base_cols, engine="pyarrow"
                    )
                except ValueError:
                    # Parquet may not have all cols; read full file as fallback
                    pairs_df = pd.read_parquet(pairs_src, engine="pyarrow")
            elif isinstance(pairs_src, pd.DataFrame):
                pairs_df = pairs_src
            else:
                raise TypeError(
                    "pairs_src must be PreferenceDataset, DataFrame, or path to parquet; "
                    f"got {type(pairs_src)}"
                )
        self.use_pair_cols = {"idx_good", "idx_bad"}.issubset(pairs_df.columns)
        if self.use_pair_cols:
            self.good_idx = torch.tensor(pairs_df["idx_good"].values, dtype=torch.long)
            self.bad_idx  = torch.tensor(pairs_df["idx_bad"].values, dtype=torch.long)
        else:
            y = torch.tensor(pairs_df["label"].values, dtype=torch.long)
            self.good_idx = torch.nonzero(y == 1, as_tuple=False).squeeze(1)
            self.bad_idx  = torch.nonzero(y == 0, as_tuple=False).squeeze(1)

        if len(self.good_idx) == 0 or len(self.bad_idx) == 0:
            raise ValueError("need at least one good and one bad sample")

        self.base_df = pairs_df.reset_index(drop=True)
        # keep a pointer to the numeric feature matrix (if available)
        self._X = self._pref_ds.X if self._pref_ds is not None else None
        # If the Parquet already contains pre‑computed vectors, cache them to avoid any per‑item work.
        if {"feat_good", "feat_bad"}.issubset(pairs_df.columns):
            self._has_precomputed = True
            # Convert list‑of‑arrays column to a single 2‑D NumPy array in one shot
            fg = np.asarray(pairs_df["feat_good"].values.tolist(), dtype=np.float32)
            fb = np.asarray(pairs_df["feat_bad"].values.tolist(),  dtype=np.float32)
            self._feat_good = torch.from_numpy(fg)
            self._feat_bad  = torch.from_numpy(fb)
            # When pre‑computed, idx_good / idx_bad refer simply to row positions
            self.good_idx = torch.arange(len(self._feat_good))
            self.bad_idx  = torch.arange(len(self._feat_bad))
            if torch.cuda.is_available():
                self._feat_good = self._feat_good.pin_memory()
                self._feat_bad  = self._feat_bad.pin_memory()
        else:
            self._has_precomputed = False

    def __len__(self):
        return max(len(self.good_idx), len(self.bad_idx))

    def __getitem__(self, _):
        """Return a random pair as tensors (good, bad, label)."""
        if self._has_precomputed:
            # Simple fast path: features are already tensors on CPU
            g_idx = int(self.good_idx[torch.randint(len(self.good_idx), (1,))])
            b_idx = int(self.bad_idx[torch.randint(len(self.bad_idx), (1,))])
            g_tensor = self._feat_good[g_idx]
            b_tensor = self._feat_bad[b_idx]
        else:
            # Existing logic (fallback)
            if self.use_pair_cols:
                g_row = self.base_df.iloc[int(self.good_idx[torch.randint(len(self.good_idx), (1,))])]
                b_row = self.base_df.iloc[int(self.bad_idx[torch.randint(len(self.bad_idx), (1,))])]
                g_idx = int(g_row["idx_good"])
                b_idx = int(b_row["idx_bad"])
            else:
                g_idx = int(self.good_idx[torch.randint(len(self.good_idx), (1,))])
                b_idx = int(self.bad_idx[torch.randint(len(self.bad_idx), (1,))])

            if self._X is None:
                g_tensor = _sanitize_features(self.base_df.iloc[[g_idx]], label_col="label", verbose=False)
                b_tensor = _sanitize_features(self.base_df.iloc[[b_idx]], label_col="label", verbose=False)
            else:
                g_tensor = self._X[g_idx]
                b_tensor = self._X[b_idx]

        label = torch.tensor(1, dtype=torch.long)
        return g_tensor, b_tensor, label


class PreferenceDataModule(pl.LightningDataModule):
    def __init__(
        self,
        pairs_file: str,
        prices_dir: str,
        lookback: int = 30,
        cache_file: str | None = None,
        batch_size: int = 256,
        num_workers: int = 4,
        val_fraction: float = 0.1,
    ):
        super().__init__()
        self.pairs_file = pairs_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.prices_dir = prices_dir
        self.lookback = lookback
        self.cache_file = cache_file

    def setup(self, stage: str | None = None):
        if self.cache_file and Path(self.cache_file).exists():
            print(f"[INFO] loading cached pairs from {self.cache_file}")
            full = CachedPairwiseDataset(self.cache_file)
        else:
            full = PreferenceDataset(self.pairs_file, self.prices_dir, lookback=self.lookback)
        val_len = max(1, int(len(full) * self.val_fraction))
        train_len = len(full) - val_len
        train_base, val_base = random_split(full, [train_len, val_len])

        # If we already loaded CachedPairwiseDataset (or PairwiseDataset), use as‑is
        def to_pair(ds):
            if isinstance(ds, torch.utils.data.Subset):
                ds = ds.dataset
            return ds if isinstance(ds, (PairwiseDataset, CachedPairwiseDataset)) else PairwiseDataset(ds)

        self.train_set = to_pair(train_base)
        self.val_set   = to_pair(val_base)
        
        # expose feature dim for model config checks
        self.num_features = full.num_features

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

# ───────────────────────── cached dataset ──────────────────────────
class CachedPairwiseDataset(Dataset):
    """Fast‑loading dataset that reads a torch‑saved cache produced by scripts/cache_pairs.py."""

    def __init__(self, cache_file: str):
        obj = torch.load(cache_file, map_location="cpu")
        self._feat_good = obj["feat_good"]  # shape (N, d)
        self._feat_bad  = obj["feat_bad"]   # shape (N, d)
        self.good_idx   = obj["good_idx"]
        self.bad_idx    = obj["bad_idx"]
        self.num_features = self._feat_good.shape[1]

        # Pin memory for faster GPU transfer
        if torch.cuda.is_available():
            self._feat_good = self._feat_good.pin_memory()
            self._feat_bad  = self._feat_bad.pin_memory()

    def __len__(self) -> int:
        return max(len(self.good_idx), len(self.bad_idx))

    def __getitem__(self, _):
        g = int(self.good_idx[torch.randint(len(self.good_idx), (1,))])
        b = int(self.bad_idx[torch.randint(len(self.bad_idx), (1,))])
        return self._feat_good[g], self._feat_bad[b], torch.tensor(1, dtype=torch.long)