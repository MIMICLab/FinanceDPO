"""Optimized backtesting with batch processing and vectorized operations."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import concurrent.futures
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import hydra
from omegaconf import DictConfig
from tqdm import tqdm


class VectorizedBacktester:
    """Vectorized backtesting engine for efficient performance evaluation."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: DictConfig,
        device: torch.device,
        batch_size: int = 32,
        n_workers: int = 4,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.lookback = config.data.lookback
        
        # Move model to device and set to eval mode
        self.model = self.model.to(device)
        self.model.eval()
        
    def extract_features_batch(
        self,
        prices_dict: Dict[str, np.ndarray],
        symbols: List[str],
        start_idx: int,
        end_idx: int,
    ) -> Tuple[torch.Tensor, List[str], List[int]]:
        """Extract features for multiple symbols in batch.
        
        Returns:
            features: Tensor of shape (n_valid, lookback-1)
            valid_symbols: List of symbols with valid data
            valid_indices: List of corresponding indices
        """
        all_features = []
        valid_symbols = []
        valid_indices = []
        
        for symbol in symbols:
            if symbol not in prices_dict:
                continue
                
            prices = prices_dict[symbol]
            
            # Vectorized feature extraction for all valid indices
            for idx in range(max(start_idx, self.lookback), min(end_idx, len(prices))):
                window = prices[idx - self.lookback + 1:idx + 1]
                if len(window) == self.lookback:
                    returns = np.diff(window) / window[:-1]
                    all_features.append(returns)
                    valid_symbols.append(symbol)
                    valid_indices.append(idx)
                    
        if not all_features:
            return torch.empty(0, self.lookback - 1), [], []
            
        features = torch.tensor(all_features, dtype=torch.float32)
        return features, valid_symbols, valid_indices
        
    def compute_scores_batch(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """Compute model scores for a batch of features.
        
        Args:
            features: Tensor of shape (batch_size, lookback-1)
            
        Returns:
            scores: Tensor of shape (batch_size,)
        """
        if len(features) == 0:
            return torch.empty(0)
            
        with torch.no_grad():
            # Process in mini-batches if needed
            if len(features) > self.batch_size:
                scores = []
                for i in range(0, len(features), self.batch_size):
                    batch = features[i:i + self.batch_size].to(self.device)
                    batch_scores = self.model(batch).cpu()
                    scores.append(batch_scores)
                return torch.cat(scores)
            else:
                features = features.to(self.device)
                return self.model(features).cpu()
                
    def load_prices_parallel(
        self,
        prices_dir: Path,
        symbols: List[str]
    ) -> Dict[str, np.ndarray]:
        """Load price data for multiple symbols in parallel."""
        prices_dict = {}
        
        def load_symbol(symbol: str) -> Tuple[str, Optional[np.ndarray]]:
            csv_path = prices_dir / f"{symbol}.csv"
            if not csv_path.exists():
                return symbol, None
            try:
                prices = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=4)
                return symbol, prices
            except Exception:
                return symbol, None
                
        # Load in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(load_symbol, sym) for sym in symbols]
            for future in concurrent.futures.as_completed(futures):
                symbol, prices = future.result()
                if prices is not None:
                    prices_dict[symbol] = prices
                    
        return prices_dict
        
    def backtest_period(
        self,
        prices_dict: Dict[str, np.ndarray],
        symbols: List[str],
        start_date: int,
        end_date: int,
        rebalance_freq: int = 20,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """Run backtest for a specific period with batch processing.
        
        Returns:
            DataFrame with daily returns
        """
        results = []
        
        # Process each rebalance period
        for date in range(start_date, end_date, rebalance_freq):
            # Extract features for all symbols at once
            features, valid_symbols, valid_indices = self.extract_features_batch(
                prices_dict, symbols, date, date + 1
            )
            
            if len(features) == 0:
                continue
                
            # Compute scores in batch
            scores = self.compute_scores_batch(features)
            
            # Create ranking DataFrame
            ranking_df = pd.DataFrame({
                "symbol": valid_symbols,
                "score": scores.numpy(),
                "date": date,
            })
            
            # Select top K symbols
            top_symbols = ranking_df.nlargest(top_k, "score")["symbol"].tolist()
            
            # Calculate returns for holding period
            for hold_day in range(rebalance_freq):
                current_date = date + hold_day
                if current_date >= end_date:
                    break
                    
                daily_returns = []
                for symbol in top_symbols:
                    if symbol not in prices_dict:
                        continue
                    prices = prices_dict[symbol]
                    if current_date + 1 < len(prices):
                        ret = (prices[current_date + 1] - prices[current_date]) / prices[current_date]
                        daily_returns.append(ret)
                        
                if daily_returns:
                    avg_return = np.mean(daily_returns)
                    results.append({
                        "date": current_date,
                        "return": avg_return,
                        "n_positions": len(daily_returns),
                    })
                    
        return pd.DataFrame(results)
        
    def calculate_metrics(self, returns_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics from returns."""
        if len(returns_df) == 0:
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
            }
            
        # Calculate cumulative returns
        returns_df["cumulative"] = (1 + returns_df["return"]).cumprod()
        
        # Metrics
        total_return = returns_df["cumulative"].iloc[-1] - 1
        
        # Sharpe ratio (annualized)
        daily_returns = returns_df["return"]
        sharpe = np.sqrt(252) * daily_returns.mean() / (daily_returns.std() + 1e-8)
        
        # Maximum drawdown
        cumulative = returns_df["cumulative"].values
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (daily_returns > 0).mean()
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "n_trades": len(returns_df),
        }


@hydra.main(version_base=None, config_path="../conf", config_name="backtest")
def main(cfg: DictConfig) -> None:
    """Run optimized backtesting."""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    checkpoint_path = Path(cfg.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize model based on type
    if "optimized" in str(checkpoint_path):
        from dpo_forecasting.models.optimized_dpo_model import OptimizedDPOModel
        model = OptimizedDPOModel(
            cfg=checkpoint["hyper_parameters"]["cfg"],
            lookback=cfg.data.lookback,
        )
    else:
        from dpo_forecasting.models.dpo_model import DPOModel
        model = DPOModel(
            cfg=checkpoint["hyper_parameters"]["cfg"],
            lookback=cfg.data.lookback,
        )
        
    # Load state dict
    state_dict = checkpoint["state_dict"]
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    
    # Initialize backtester
    backtester = VectorizedBacktester(
        model=model,
        config=cfg,
        device=device,
        batch_size=cfg.get("batch_size", 64),
        n_workers=cfg.get("n_workers", 4),
    )
    
    # Load symbols
    prices_dir = Path(cfg.data.prices_dir)
    if cfg.get("symbols"):
        symbols = cfg.symbols
    else:
        # Use all available symbols
        symbols = [f.stem for f in prices_dir.glob("*.csv")]
        
    print(f"Processing {len(symbols)} symbols")
    
    # Load all price data in parallel
    print("Loading price data...")
    prices_dict = backtester.load_prices_parallel(prices_dir, symbols)
    print(f"Loaded data for {len(prices_dict)} symbols")
    
    # Run backtest
    print("Running backtest...")
    start_date = cfg.get("start_date", 252)  # Skip first year for warmup
    end_date = cfg.get("end_date", 2520)     # ~10 years
    
    returns_df = backtester.backtest_period(
        prices_dict=prices_dict,
        symbols=list(prices_dict.keys()),
        start_date=start_date,
        end_date=end_date,
        rebalance_freq=cfg.get("rebalance_freq", 20),
        top_k=cfg.get("top_k", 10),
    )
    
    # Calculate and display metrics
    metrics = backtester.calculate_metrics(returns_df)
    
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Number of Trades: {metrics['n_trades']}")
    
    # Save results
    if cfg.get("save_results", True):
        output_dir = Path("backtest_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save returns
        returns_df.to_csv(output_dir / "returns.csv", index=False)
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(output_dir / "metrics.csv", index=False)
        
        print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()