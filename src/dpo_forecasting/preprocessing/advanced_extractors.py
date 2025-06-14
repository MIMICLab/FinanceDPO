# ─── src/dpo_forecasting/preprocessing/advanced_extractors.py ─────────────
"""
Advanced feature extractors for enhanced DPO financial forecasting.

Includes technical indicators, market microstructure features, and 
cross-sectional information.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats


class AdvancedFeatureExtractor:
    """Extract comprehensive features for financial time series.
    
    Features include:
    - Price-based: returns, log returns, price ratios
    - Volume-based: volume ratios, volume-weighted returns
    - Volatility: realized vol, GARCH-like features
    - Technical indicators: RSI, MACD, Bollinger bands
    - Market microstructure: bid-ask proxies, roll measure
    - Cross-sectional: market beta, sector relative performance
    """
    
    def __init__(
        self,
        lookback: int = 256,
        short_window: int = 20,
        long_window: int = 50,
        vol_window: int = 20,
        use_volume: bool = True,
        use_technical: bool = True,
        use_microstructure: bool = True,
    ):
        self.lookback = lookback
        self.short_window = short_window
        self.long_window = long_window
        self.vol_window = vol_window
        self.use_volume = use_volume
        self.use_technical = use_technical
        self.use_microstructure = use_microstructure
        
    def __call__(
        self,
        data: pd.DataFrame,
        idx: int,
        market_data: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """Extract features at given index.
        
        Parameters
        ----------
        data : pd.DataFrame
            Must contain columns: Close, Volume, High, Low, Open
        idx : int
            Current index (must be >= lookback)
        market_data : pd.DataFrame, optional
            Market index data for relative features
            
        Returns
        -------
        numpy.ndarray
            Feature vector
        """
        if idx < self.lookback:
            raise IndexError(f"idx {idx} must be >= lookback {self.lookback}")
            
        # Get window
        window = data.iloc[idx - self.lookback:idx]
        
        features = []
        
        # 1. Price-based features
        close = window['Close'].values
        returns = self._compute_returns(close)
        features.extend([
            returns[-1],  # latest return
            returns[-5:].mean(),  # 5-day return
            returns[-20:].mean(),  # 20-day return
            returns.mean(),  # mean return
            returns.std(),  # volatility
            stats.skew(returns),  # skewness
            stats.kurtosis(returns),  # kurtosis
            np.log(close[-1] / close[0]),  # log return over window
        ])
        
        # Moving averages
        ma_short = close[-self.short_window:].mean()
        ma_long = close[-self.long_window:].mean()
        features.extend([
            close[-1] / ma_short - 1,  # price to short MA
            close[-1] / ma_long - 1,   # price to long MA
            ma_short / ma_long - 1,     # MA crossover signal
        ])
        
        # 2. Volume features
        if self.use_volume and 'Volume' in window.columns:
            volume = window['Volume'].values
            features.extend(self._extract_volume_features(close, volume))
            
        # 3. Technical indicators
        if self.use_technical:
            features.extend(self._extract_technical_features(window))
            
        # 4. Microstructure features
        if self.use_microstructure and all(col in window.columns for col in ['High', 'Low']):
            features.extend(self._extract_microstructure_features(window))
            
        # 5. Cross-sectional features
        if market_data is not None:
            features.extend(self._extract_cross_sectional_features(
                window, market_data.iloc[idx - self.lookback:idx]
            ))
            
        return np.array(features, dtype=np.float32)
    
    def _compute_returns(self, prices: np.ndarray) -> np.ndarray:
        """Compute simple returns."""
        return (prices[1:] - prices[:-1]) / prices[:-1]
    
    def _extract_volume_features(
        self, 
        prices: np.ndarray, 
        volumes: np.ndarray
    ) -> List[float]:
        """Extract volume-based features."""
        returns = self._compute_returns(prices)
        volumes = volumes[1:]  # align with returns
        
        # Volume ratios
        vol_ma = volumes[-20:].mean()
        vol_ratio = volumes[-1] / (vol_ma + 1e-8)
        
        # Volume-weighted returns
        vwap = np.sum(prices[-20:] * volumes[-20:]) / (np.sum(volumes[-20:]) + 1e-8)
        
        # Amihud illiquidity
        illiquidity = np.mean(np.abs(returns[-20:]) / (volumes[-20:] + 1e-8))
        
        return [
            vol_ratio,
            prices[-1] / vwap - 1,
            np.log(illiquidity + 1e-8),
            np.corrcoef(returns[-20:], volumes[-20:])[0, 1],  # return-volume correlation
        ]
    
    def _extract_technical_features(self, window: pd.DataFrame) -> List[float]:
        """Extract technical indicators."""
        close = window['Close'].values
        high = window['High'].values
        low = window['Low'].values
        
        # RSI
        rsi = self._compute_rsi(close, 14)
        
        # MACD
        ema_12 = self._ema(close, 12)
        ema_26 = self._ema(close, 26)
        macd = ema_12 - ema_26
        signal = self._ema(macd, 9)
        
        # Bollinger Bands
        ma = close[-20:].mean()
        std = close[-20:].std()
        bb_upper = ma + 2 * std
        bb_lower = ma - 2 * std
        bb_width = (bb_upper - bb_lower) / ma
        bb_position = (close[-1] - bb_lower) / (bb_upper - bb_lower + 1e-8)
        
        # ATR (Average True Range)
        tr = np.maximum.reduce([
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        ])
        atr = tr[-14:].mean()
        
        return [
            rsi,
            macd[-1] / close[-1],  # normalized MACD
            (macd[-1] - signal[-1]) / close[-1],  # MACD histogram
            bb_width,
            bb_position,
            atr / close[-1],  # normalized ATR
        ]
    
    def _extract_microstructure_features(self, window: pd.DataFrame) -> List[float]:
        """Extract market microstructure features."""
        high = window['High'].values
        low = window['Low'].values
        close = window['Close'].values
        open_ = window['Open'].values
        
        # Garman-Klass volatility estimator
        gk_vol = np.sqrt(np.mean(
            0.5 * np.log(high / low) ** 2 - 
            (2 * np.log(2) - 1) * np.log(close / open_) ** 2
        ))
        
        # Parkinson volatility
        park_vol = np.sqrt(np.mean(np.log(high / low) ** 2) / (4 * np.log(2)))
        
        # Roll's implicit spread estimator
        returns = self._compute_returns(close)
        roll_spread = 2 * np.sqrt(-np.cov(returns[:-1], returns[1:])[0, 1])
        
        # Intraday momentum
        intraday_ret = (close - open_) / open_
        
        return [
            gk_vol,
            park_vol,
            roll_spread / close[-1],  # normalized spread
            intraday_ret[-1],
            intraday_ret[-5:].mean(),
        ]
    
    def _extract_cross_sectional_features(
        self, 
        stock_window: pd.DataFrame,
        market_window: pd.DataFrame
    ) -> List[float]:
        """Extract cross-sectional features relative to market."""
        stock_returns = self._compute_returns(stock_window['Close'].values)
        market_returns = self._compute_returns(market_window['Close'].values)
        
        # Beta
        cov_matrix = np.cov(stock_returns[-60:], market_returns[-60:])
        beta = cov_matrix[0, 1] / (cov_matrix[1, 1] + 1e-8)
        
        # Relative strength
        stock_perf = stock_window['Close'].iloc[-1] / stock_window['Close'].iloc[-20] - 1
        market_perf = market_window['Close'].iloc[-1] / market_window['Close'].iloc[-20] - 1
        rel_strength = stock_perf - market_perf
        
        # Correlation
        correlation = np.corrcoef(stock_returns[-20:], market_returns[-20:])[0, 1]
        
        return [beta, rel_strength, correlation]
    
    def _compute_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Compute Relative Strength Index."""
        returns = self._compute_returns(prices)
        gains = np.where(returns > 0, returns, 0)
        losses = np.where(returns < 0, -returns, 0)
        
        avg_gain = gains[-period:].mean()
        avg_loss = losses[-period:].mean()
        
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Exponential moving average."""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features."""
        names = [
            'return_1d', 'return_5d', 'return_20d', 'return_mean', 'return_std',
            'return_skew', 'return_kurtosis', 'log_return_window',
            'price_to_ma_short', 'price_to_ma_long', 'ma_crossover'
        ]
        
        if self.use_volume:
            names.extend(['volume_ratio', 'price_to_vwap', 'log_illiquidity', 'return_volume_corr'])
            
        if self.use_technical:
            names.extend(['rsi', 'macd_norm', 'macd_hist_norm', 'bb_width', 'bb_position', 'atr_norm'])
            
        if self.use_microstructure:
            names.extend(['gk_vol', 'park_vol', 'roll_spread_norm', 'intraday_ret', 'intraday_ret_5d'])
            
        return names
    
    def __getstate__(self) -> Dict[str, Any]:
        """For pickling."""
        return self.__dict__.copy()
    
    def __setstate__(self, state: Dict[str, Any]) -> None:
        """For unpickling."""
        self.__dict__.update(state)


class AdaptiveWindowExtractor:
    """Feature extractor with adaptive lookback windows based on market regime."""
    
    def __init__(
        self,
        min_lookback: int = 20,
        max_lookback: int = 252,
        vol_threshold_low: float = 0.1,
        vol_threshold_high: float = 0.3,
    ):
        self.min_lookback = min_lookback
        self.max_lookback = max_lookback
        self.vol_threshold_low = vol_threshold_low
        self.vol_threshold_high = vol_threshold_high
        self.base_extractor = AdvancedFeatureExtractor()
        
    def __call__(
        self,
        data: pd.DataFrame,
        idx: int,
        market_data: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """Extract features with adaptive window."""
        # Estimate recent volatility
        recent_vol = self._estimate_volatility(data, idx)
        
        # Determine adaptive window size
        if recent_vol < self.vol_threshold_low:
            # Low volatility: use longer window
            lookback = self.max_lookback
        elif recent_vol > self.vol_threshold_high:
            # High volatility: use shorter window
            lookback = self.min_lookback
        else:
            # Linear interpolation
            ratio = (recent_vol - self.vol_threshold_low) / (self.vol_threshold_high - self.vol_threshold_low)
            lookback = int(self.max_lookback - ratio * (self.max_lookback - self.min_lookback))
            
        # Update base extractor lookback
        self.base_extractor.lookback = lookback
        
        # Extract features
        features = self.base_extractor(data, idx, market_data)
        
        # Add regime indicators
        regime_features = np.array([
            recent_vol,
            lookback / self.max_lookback,  # normalized window size
        ], dtype=np.float32)
        
        return np.concatenate([features, regime_features])
    
    def _estimate_volatility(self, data: pd.DataFrame, idx: int) -> float:
        """Estimate recent volatility."""
        if idx < 20:
            return 0.15  # default
            
        recent_close = data['Close'].iloc[idx-20:idx].values
        returns = (recent_close[1:] - recent_close[:-1]) / recent_close[:-1]
        return returns.std() * np.sqrt(252)  # annualized


__all__ = [
    "AdvancedFeatureExtractor",
    "AdaptiveWindowExtractor",
]