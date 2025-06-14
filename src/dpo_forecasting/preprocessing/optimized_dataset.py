"""Optimized Dataset with prefetching, caching, and vectorized operations."""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import concurrent.futures
import threading
import queue

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torch.utils.data._utils.collate import default_collate


class OptimizedPairwiseDataset(Dataset):
    """Optimized dataset with memory-mapped arrays and efficient sampling."""
    
    def __init__(
        self,
        feat_good: torch.Tensor,
        feat_bad: torch.Tensor,
        good_idx: torch.Tensor,
        bad_idx: torch.Tensor,
        prefetch_size: int = 1000,
        device: Optional[torch.device] = None,
    ):
        self.feat_good = feat_good
        self.feat_bad = feat_bad
        self.good_idx = good_idx
        self.bad_idx = bad_idx
        self.prefetch_size = prefetch_size
        self.device = device or torch.device("cpu")
        
        # Pre-generate random indices for faster sampling
        self.n_good = len(good_idx)
        self.n_bad = len(bad_idx)
        self._regenerate_indices()
        
        # Pin memory if using CUDA
        if torch.cuda.is_available() and self.device.type == "cuda":
            self.feat_good = self.feat_good.pin_memory()
            self.feat_bad = self.feat_bad.pin_memory()
            
    def _regenerate_indices(self):
        """Pre-generate random indices for an epoch."""
        self.good_samples = torch.randint(0, self.n_good, (len(self) + self.prefetch_size,))
        self.bad_samples = torch.randint(0, self.n_bad, (len(self) + self.prefetch_size,))
        self.sample_counter = 0
        
    def __len__(self):
        return max(self.n_good, self.n_bad)
    
    def __getitem__(self, idx):
        # Use pre-generated indices
        if self.sample_counter >= len(self.good_samples) - 1:
            self._regenerate_indices()
            
        g_idx = self.good_idx[self.good_samples[self.sample_counter]]
        b_idx = self.bad_idx[self.bad_samples[self.sample_counter]]
        self.sample_counter += 1
        
        g_tensor = self.feat_good[g_idx]
        b_tensor = self.feat_bad[b_idx]
        label = torch.tensor(1, dtype=torch.long)
        
        return g_tensor, b_tensor, label


class VectorizedFeatureExtractor:
    """Vectorized feature extraction for batch processing."""
    
    def __init__(self, lookback: int = 31):
        self.lookback = lookback
        
    def extract_batch(
        self,
        prices: np.ndarray,
        indices: np.ndarray
    ) -> np.ndarray:
        """Extract features for multiple indices at once.
        
        Args:
            prices: Array of shape (n_timesteps,)
            indices: Array of indices to extract features for
            
        Returns:
            Features array of shape (len(indices), lookback-1)
        """
        valid_mask = indices >= self.lookback
        valid_indices = indices[valid_mask]
        
        if len(valid_indices) == 0:
            return np.array([])
            
        # Vectorized return calculation
        features = np.zeros((len(valid_indices), self.lookback - 1))
        
        for i, idx in enumerate(valid_indices):
            window = prices[idx - self.lookback + 1:idx + 1]
            if len(window) == self.lookback:
                returns = np.diff(window) / window[:-1]
                features[i] = returns
                
        return features


class CachedDataLoader:
    """Data loader with background prefetching and caching."""
    
    def __init__(
        self,
        prices_dir: str,
        cache_size: int = 100,
        n_workers: int = 4
    ):
        self.prices_dir = Path(prices_dir)
        self.cache = {}
        self.cache_lock = threading.Lock()
        self.cache_size = cache_size
        self.n_workers = n_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=n_workers)
        self.prefetch_queue = queue.Queue(maxsize=50)
        
    def load_prices(self, symbol: str) -> Optional[np.ndarray]:
        """Load prices with caching."""
        with self.cache_lock:
            if symbol in self.cache:
                return self.cache[symbol]
                
        csv_path = self.prices_dir / f"{symbol}.csv"
        if not csv_path.exists():
            return None
            
        try:
            # Load only close prices (column 4)
            prices = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=4)
            
            # Update cache
            with self.cache_lock:
                if len(self.cache) >= self.cache_size:
                    # Simple LRU: remove first item
                    self.cache.pop(next(iter(self.cache)))
                self.cache[symbol] = prices
                
            return prices
        except Exception:
            return None
            
    def prefetch_symbols(self, symbols: List[str]):
        """Prefetch multiple symbols in background."""
        futures = []
        for symbol in symbols:
            if symbol not in self.cache:
                future = self.executor.submit(self.load_prices, symbol)
                futures.append(future)
        return futures


def create_optimized_dataset(
    pairs_file: str,
    prices_dir: str,
    lookback: int = 31,
    cache_file: Optional[str] = None,
    batch_process: bool = True,
    n_workers: int = 4,
) -> OptimizedPairwiseDataset:
    """Create optimized dataset with vectorized operations.
    
    Args:
        pairs_file: Path to pairs parquet file
        prices_dir: Directory containing price CSV files
        lookback: Number of days for feature window
        cache_file: Optional path to save/load cached features
        batch_process: Whether to use batch processing
        n_workers: Number of workers for parallel processing
        
    Returns:
        OptimizedPairwiseDataset instance
    """
    # Try to load from cache first
    if cache_file and Path(cache_file).exists():
        print(f"Loading cached dataset from {cache_file}")
        data = torch.load(cache_file, map_location="cpu")
        return OptimizedPairwiseDataset(
            feat_good=data["feat_good"],
            feat_bad=data["feat_bad"],
            good_idx=data["good_idx"],
            bad_idx=data["bad_idx"],
        )
    
    # Load pairs data
    df = pd.read_parquet(pairs_file)
    
    # Initialize components
    extractor = VectorizedFeatureExtractor(lookback)
    loader = CachedDataLoader(prices_dir, n_workers=n_workers)
    
    # Group by symbol for batch processing
    symbol_groups = df.groupby("symbol")
    
    all_features = []
    all_labels = []
    valid_indices = []
    
    # Process symbols in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        
        for symbol, group in symbol_groups:
            future = executor.submit(
                process_symbol_group,
                symbol,
                group,
                loader,
                extractor,
                lookback
            )
            futures[future] = symbol
            
        # Collect results
        for future in concurrent.futures.as_completed(futures):
            features, labels, indices = future.result()
            if features is not None:
                all_features.extend(features)
                all_labels.extend(labels)
                valid_indices.extend(indices)
    
    # Convert to tensors
    feat_tensor = torch.tensor(np.array(all_features), dtype=torch.float32)
    label_tensor = torch.tensor(all_labels, dtype=torch.long)
    
    # Separate good and bad indices
    good_mask = label_tensor == 1
    good_idx = torch.where(good_mask)[0]
    bad_idx = torch.where(~good_mask)[0]
    
    # Create dataset
    dataset = OptimizedPairwiseDataset(
        feat_good=feat_tensor,
        feat_bad=feat_tensor,
        good_idx=good_idx,
        bad_idx=bad_idx,
    )
    
    # Save cache if requested
    if cache_file:
        print(f"Saving dataset cache to {cache_file}")
        torch.save({
            "feat_good": feat_tensor,
            "feat_bad": feat_tensor,
            "good_idx": good_idx,
            "bad_idx": bad_idx,
        }, cache_file)
    
    return dataset


def process_symbol_group(
    symbol: str,
    group: pd.DataFrame,
    loader: CachedDataLoader,
    extractor: VectorizedFeatureExtractor,
    lookback: int
) -> Tuple[Optional[List[np.ndarray]], Optional[List[int]], Optional[List[int]]]:
    """Process a group of pairs for a single symbol."""
    prices = loader.load_prices(symbol)
    if prices is None:
        return None, None, None
        
    # Extract indices
    indices = group["idx_good"].values if "idx_good" in group else group.index.values
    
    # Vectorized feature extraction
    features = extractor.extract_batch(prices, indices)
    
    if len(features) == 0:
        return None, None, None
        
    labels = group["label"].values[:len(features)]
    valid_indices = list(range(len(features)))
    
    return features.tolist(), labels.tolist(), valid_indices


class OptimizedDataModule(pl.LightningDataModule):
    """Optimized data module with prefetching and caching."""
    
    def __init__(
        self,
        pairs_file: str,
        prices_dir: str,
        lookback: int = 30,
        cache_file: Optional[str] = None,
        batch_size: int = 256,
        num_workers: int = 4,
        val_fraction: float = 0.1,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
    ):
        super().__init__()
        self.pairs_file = pairs_file
        self.prices_dir = prices_dir
        self.lookback = lookback
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers and num_workers > 0
        
    def setup(self, stage: Optional[str] = None):
        # Create optimized dataset
        full_dataset = create_optimized_dataset(
            self.pairs_file,
            self.prices_dir,
            self.lookback,
            self.cache_file,
            n_workers=self.num_workers,
        )
        
        # Split into train/val
        val_len = max(1, int(len(full_dataset) * self.val_fraction))
        train_len = len(full_dataset) - val_len
        
        # Use generator for reproducible splits
        generator = torch.Generator().manual_seed(42)
        train_indices, val_indices = torch.utils.data.random_split(
            range(len(full_dataset)),
            [train_len, val_len],
            generator=generator
        )
        
        # Create subset datasets
        self.train_dataset = torch.utils.data.Subset(full_dataset, train_indices.indices)
        self.val_dataset = torch.utils.data.Subset(full_dataset, val_indices.indices)
        
        # Store feature dimension
        self.num_features = full_dataset.feat_good.shape[1]
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )