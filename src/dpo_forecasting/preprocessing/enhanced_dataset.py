# ─── src/dpo_forecasting/preprocessing/enhanced_dataset.py ────────────────
"""
Enhanced dataset classes with advanced sampling strategies and data augmentation.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Callable
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .advanced_extractors import AdvancedFeatureExtractor, AdaptiveWindowExtractor


class EnhancedPreferenceDataset(Dataset):
    """Enhanced dataset with advanced preference pair generation.
    
    Features:
    - Dynamic preference thresholds based on market regime
    - Balanced sampling across different market conditions
    - Data augmentation techniques
    - Multi-timeframe preferences
    """
    
    def __init__(
        self,
        data_dir: Path,
        symbols: List[str],
        extractor: Optional[Callable] = None,
        forward_days: int = 8,
        preference_quantiles: Tuple[float, float] = (0.2, 0.8),
        lookback: int = 256,
        use_adaptive_thresholds: bool = True,
        augment_data: bool = True,
        cache_features: bool = True,
        market_data_path: Optional[Path] = None,
    ):
        self.data_dir = Path(data_dir)
        self.symbols = symbols
        self.extractor = extractor or AdvancedFeatureExtractor(lookback=lookback)
        self.forward_days = forward_days
        self.preference_quantiles = preference_quantiles
        self.use_adaptive_thresholds = use_adaptive_thresholds
        self.augment_data = augment_data
        self.cache_features = cache_features
        
        # Load market data if provided
        self.market_data = None
        if market_data_path and market_data_path.exists():
            self.market_data = pd.read_csv(market_data_path, parse_dates=['Date'])
            self.market_data.set_index('Date', inplace=True)
        
        # Storage
        self.preference_pairs = []
        self.feature_cache = {}
        
        # Generate preference pairs
        self._generate_preference_pairs()
        
    def _generate_preference_pairs(self):
        """Generate preference pairs with enhanced logic."""
        print("Generating enhanced preference pairs...")
        
        for symbol in self.symbols:
            csv_path = self.data_dir / f"{symbol}.csv"
            if not csv_path.exists():
                continue
                
            # Load data
            df = pd.read_csv(csv_path, parse_dates=['Date'])
            df.set_index('Date', inplace=True)
            
            # Calculate forward returns
            forward_returns = self._calculate_forward_returns(df)
            
            # Determine thresholds
            if self.use_adaptive_thresholds:
                thresholds = self._calculate_adaptive_thresholds(df, forward_returns)
            else:
                low_q, high_q = self.preference_quantiles
                thresholds = [(
                    np.quantile(forward_returns, low_q),
                    np.quantile(forward_returns, high_q)
                )]
            
            # Generate pairs for each threshold set
            for low_thresh, high_thresh in thresholds:
                pairs = self._generate_pairs_for_threshold(
                    df, forward_returns, low_thresh, high_thresh, symbol
                )
                self.preference_pairs.extend(pairs)
                
        # Shuffle pairs
        random.shuffle(self.preference_pairs)
        print(f"Generated {len(self.preference_pairs)} preference pairs")
        
    def _calculate_forward_returns(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate forward-looking returns."""
        close = df['Close'].values
        forward_ret = np.zeros(len(close))
        
        for i in range(len(close) - self.forward_days):
            forward_ret[i] = (close[i + self.forward_days] - close[i]) / close[i]
            
        return forward_ret
    
    def _calculate_adaptive_thresholds(
        self,
        df: pd.DataFrame,
        forward_returns: np.ndarray
    ) -> List[Tuple[float, float]]:
        """Calculate adaptive thresholds based on market regimes."""
        # Simple regime detection based on volatility
        returns = df['Close'].pct_change().values
        vol_window = 20
        
        thresholds = []
        
        # Low volatility regime
        low_vol_mask = pd.Series(returns).rolling(vol_window).std() < 0.01
        low_vol_returns = forward_returns[low_vol_mask.values[:-self.forward_days]]
        if len(low_vol_returns) > 100:
            thresholds.append((
                np.quantile(low_vol_returns, 0.3),
                np.quantile(low_vol_returns, 0.7)
            ))
        
        # Normal regime
        thresholds.append((
            np.quantile(forward_returns, self.preference_quantiles[0]),
            np.quantile(forward_returns, self.preference_quantiles[1])
        ))
        
        # High volatility regime
        high_vol_mask = pd.Series(returns).rolling(vol_window).std() > 0.03
        high_vol_returns = forward_returns[high_vol_mask.values[:-self.forward_days]]
        if len(high_vol_returns) > 100:
            thresholds.append((
                np.quantile(high_vol_returns, 0.1),
                np.quantile(high_vol_returns, 0.9)
            ))
            
        return thresholds
    
    def _generate_pairs_for_threshold(
        self,
        df: pd.DataFrame,
        forward_returns: np.ndarray,
        low_thresh: float,
        high_thresh: float,
        symbol: str
    ) -> List[Dict]:
        """Generate preference pairs for given thresholds."""
        pairs = []
        
        # Find good and bad indices
        good_indices = np.where(forward_returns > high_thresh)[0]
        bad_indices = np.where(forward_returns < low_thresh)[0]
        
        # Filter valid indices
        min_idx = self.extractor.lookback if hasattr(self.extractor, 'lookback') else 256
        good_indices = good_indices[good_indices >= min_idx]
        bad_indices = bad_indices[bad_indices >= min_idx]
        
        # Limit pairs per symbol to avoid imbalance
        max_pairs_per_symbol = min(len(good_indices), len(bad_indices), 1000)
        
        # Sample pairs
        for _ in range(max_pairs_per_symbol):
            good_idx = random.choice(good_indices)
            bad_idx = random.choice(bad_indices)
            
            # Create pair
            pair = {
                'symbol': symbol,
                'good_idx': good_idx,
                'bad_idx': bad_idx,
                'good_return': forward_returns[good_idx],
                'bad_return': forward_returns[bad_idx],
                'df': df,  # Store reference for feature extraction
            }
            
            # Data augmentation
            if self.augment_data:
                augmented_pairs = self._augment_pair(pair)
                pairs.extend(augmented_pairs)
            else:
                pairs.append(pair)
                
        return pairs
    
    def _augment_pair(self, pair: Dict) -> List[Dict]:
        """Apply data augmentation to preference pair."""
        augmented = [pair]  # Original pair
        
        # Temporal jittering - slightly shift the indices
        if random.random() < 0.3:
            jitter = random.randint(-2, 2)
            if (pair['good_idx'] + jitter >= self.extractor.lookback and 
                pair['bad_idx'] + jitter >= self.extractor.lookback):
                augmented.append({
                    **pair,
                    'good_idx': pair['good_idx'] + jitter,
                    'bad_idx': pair['bad_idx'] + jitter,
                })
        
        return augmented
    
    def _extract_features(self, df: pd.DataFrame, idx: int, symbol: str) -> torch.Tensor:
        """Extract features with caching."""
        cache_key = f"{symbol}_{idx}"
        
        if self.cache_features and cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
            
        # Extract features
        if self.market_data is not None:
            # Align market data with stock data
            market_aligned = self.market_data.reindex(df.index)
            features = self.extractor(df, idx, market_aligned)
        else:
            features = self.extractor(df, idx)
            
        features_tensor = torch.from_numpy(features).float()
        
        if self.cache_features:
            self.feature_cache[cache_key] = features_tensor
            
        return features_tensor
    
    def __len__(self) -> int:
        return len(self.preference_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a preference pair."""
        pair = self.preference_pairs[idx]
        
        # Extract features
        good_features = self._extract_features(
            pair['df'], pair['good_idx'], pair['symbol']
        )
        bad_features = self._extract_features(
            pair['df'], pair['bad_idx'], pair['symbol']
        )
        
        return {
            'good_features': good_features,
            'bad_features': bad_features,
            'good_return': torch.tensor(pair['good_return'], dtype=torch.float32),
            'bad_return': torch.tensor(pair['bad_return'], dtype=torch.float32),
        }


class SequentialMarketDataset(Dataset):
    """Dataset for sequential market prediction (non-preference based).
    
    Useful for pre-training or auxiliary tasks.
    """
    
    def __init__(
        self,
        data_dir: Path,
        symbols: List[str],
        extractor: Optional[Callable] = None,
        sequence_length: int = 30,
        prediction_horizon: int = 5,
        stride: int = 1,
    ):
        self.data_dir = Path(data_dir)
        self.symbols = symbols
        self.extractor = extractor or AdvancedFeatureExtractor()
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.stride = stride
        
        self.sequences = []
        self._generate_sequences()
        
    def _generate_sequences(self):
        """Generate sequential samples."""
        for symbol in self.symbols:
            csv_path = self.data_dir / f"{symbol}.csv"
            if not csv_path.exists():
                continue
                
            df = pd.read_csv(csv_path, parse_dates=['Date'])
            df.set_index('Date', inplace=True)
            
            # Generate sequences
            min_idx = max(self.extractor.lookback, self.sequence_length)
            max_idx = len(df) - self.prediction_horizon
            
            for i in range(min_idx, max_idx, self.stride):
                self.sequences.append({
                    'symbol': symbol,
                    'start_idx': i - self.sequence_length,
                    'end_idx': i,
                    'target_idx': i + self.prediction_horizon,
                    'df': df,
                })
                
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sequence sample."""
        seq = self.sequences[idx]
        df = seq['df']
        
        # Extract features for sequence
        features = []
        for i in range(seq['start_idx'], seq['end_idx']):
            if i >= self.extractor.lookback:
                feat = self.extractor(df, i)
                features.append(feat)
                
        features = torch.stack([torch.from_numpy(f) for f in features]).float()
        
        # Target return
        current_price = df['Close'].iloc[seq['end_idx'] - 1]
        future_price = df['Close'].iloc[seq['target_idx'] - 1]
        target_return = (future_price - current_price) / current_price
        
        return {
            'features': features,
            'target_return': torch.tensor(target_return, dtype=torch.float32),
        }


__all__ = [
    "EnhancedPreferenceDataset",
    "SequentialMarketDataset",
]