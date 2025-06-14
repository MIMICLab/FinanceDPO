"""Enhanced backtesting system with advanced risk management and portfolio optimization.

Features:
- Dynamic position sizing based on volatility and risk
- Portfolio-level risk constraints
- Transaction cost modeling
- Multiple strategy variants
- Comprehensive performance metrics
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from dpo_forecasting.models.enhanced_dpo_model import EnhancedDPOModel, LightweightDPOModel
from dpo_forecasting.preprocessing.advanced_extractors import (
    AdvancedFeatureExtractor, AdaptiveWindowExtractor
)
from dpo_forecasting.utils.device import get_device


class RiskManager:
    """Portfolio risk management system."""
    
    def __init__(
        self,
        max_position_size: float = 0.1,
        max_leverage: float = 1.0,
        vol_target: float = 0.15,
        max_drawdown: float = 0.2,
        lookback: int = 60,
    ):
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.vol_target = vol_target
        self.max_drawdown = max_drawdown
        self.lookback = lookback
        
        # Track portfolio metrics
        self.positions = {}
        self.equity_curve = []
        self.peak_equity = 1.0
        
    def calculate_position_size(
        self,
        symbol: str,
        signal_strength: float,
        recent_returns: np.ndarray,
        current_positions: Dict[str, float],
    ) -> float:
        """Calculate position size with risk constraints."""
        # Base position from signal
        base_position = signal_strength * self.max_position_size
        
        # Volatility adjustment
        if len(recent_returns) > 10:
            vol = recent_returns.std() * np.sqrt(252)
            vol_scalar = min(1.0, self.vol_target / (vol + 1e-8))
            base_position *= vol_scalar
            
        # Drawdown adjustment
        if len(self.equity_curve) > 0:
            current_dd = (self.peak_equity - self.equity_curve[-1]) / self.peak_equity
            if current_dd > self.max_drawdown * 0.5:
                dd_scalar = 1.0 - (current_dd / self.max_drawdown)
                base_position *= max(0.2, dd_scalar)
                
        # Portfolio concentration limit
        total_exposure = sum(abs(p) for p in current_positions.values())
        if total_exposure + abs(base_position) > self.max_leverage:
            base_position *= (self.max_leverage - total_exposure) / abs(base_position)
            
        return np.clip(base_position, -self.max_position_size, self.max_position_size)
    
    def update_equity(self, equity: float):
        """Update equity tracking."""
        self.equity_curve.append(equity)
        self.peak_equity = max(self.peak_equity, equity)


class EnhancedBacktester:
    """Enhanced backtesting engine."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        extractor: AdvancedFeatureExtractor,
        risk_manager: RiskManager,
        transaction_cost_bps: float = 10.0,
        slippage_bps: float = 5.0,
    ):
        self.model = model
        self.extractor = extractor
        self.risk_manager = risk_manager
        self.transaction_cost_bps = transaction_cost_bps
        self.slippage_bps = slippage_bps
        
        self.model.eval()
        self.device = next(model.parameters()).device
        
    def backtest(
        self,
        data_dir: Path,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_capital: float = 1000000,
        rebalance_freq: str = 'daily',
        strategy: str = 'threshold',
    ) -> Dict:
        """Run backtest on multiple symbols.
        
        Parameters
        ----------
        data_dir : Path
            Directory containing price CSVs
        symbols : List[str]
            Symbols to trade
        start_date : str, optional
            Start date (YYYY-MM-DD)
        end_date : str, optional
            End date (YYYY-MM-DD)
        initial_capital : float
            Starting capital
        rebalance_freq : str
            Rebalancing frequency ('daily', 'weekly', 'monthly')
        strategy : str
            Trading strategy ('threshold', 'ranking', 'ml_signals')
        """
        # Load and align data
        symbol_data = self._load_and_align_data(data_dir, symbols, start_date, end_date)
        
        if not symbol_data:
            raise ValueError("No valid data found")
            
        # Get trading dates
        all_dates = sorted(set().union(*[set(df.index) for df in symbol_data.values()]))
        trading_dates = pd.DatetimeIndex(all_dates)
        
        # Initialize portfolio
        portfolio = {
            'cash': initial_capital,
            'positions': {symbol: 0 for symbol in symbols},
            'values': [],
            'dates': [],
            'trades': [],
        }
        
        # Market data for relative features
        market_data = None
        market_path = data_dir / 'SPY.csv'
        if market_path.exists():
            market_data = pd.read_csv(market_path, parse_dates=['Date'])
            market_data.set_index('Date', inplace=True)
            
        # Run backtest
        print(f"Running backtest from {trading_dates[0]} to {trading_dates[-1]}")
        
        for i, date in enumerate(tqdm(trading_dates, desc="Backtesting")):
            # Get current positions value
            position_value = self._calculate_position_value(
                portfolio['positions'], symbol_data, date
            )
            total_value = portfolio['cash'] + position_value
            
            # Update risk manager
            self.risk_manager.update_equity(total_value)
            
            # Check rebalancing
            if self._should_rebalance(date, rebalance_freq):
                # Generate signals
                signals = self._generate_signals(
                    symbol_data, date, market_data, strategy
                )
                
                # Calculate target positions
                target_positions = self._calculate_target_positions(
                    signals, symbol_data, date, portfolio['positions']
                )
                
                # Execute trades
                trades = self._execute_trades(
                    portfolio, target_positions, symbol_data, date
                )
                portfolio['trades'].extend(trades)
                
            # Record portfolio value
            portfolio['values'].append(total_value)
            portfolio['dates'].append(date)
            
        # Calculate performance metrics
        results = self._calculate_metrics(portfolio, initial_capital)
        results['portfolio'] = portfolio
        
        return results
    
    def _load_and_align_data(
        self,
        data_dir: Path,
        symbols: List[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> Dict[str, pd.DataFrame]:
        """Load and align price data."""
        symbol_data = {}
        
        for symbol in symbols:
            csv_path = data_dir / f"{symbol}.csv"
            if not csv_path.exists():
                print(f"Warning: {csv_path} not found")
                continue
                
            df = pd.read_csv(csv_path, parse_dates=['Date'])
            df.set_index('Date', inplace=True)
            
            # Filter dates
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
                
            if len(df) > self.extractor.lookback:
                symbol_data[symbol] = df
                
        return symbol_data
    
    def _should_rebalance(self, date: pd.Timestamp, freq: str) -> bool:
        """Check if we should rebalance on this date."""
        if freq == 'daily':
            return True
        elif freq == 'weekly':
            return date.weekday() == 0  # Monday
        elif freq == 'monthly':
            return date.day == 1
        return False
    
    def _generate_signals(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        date: pd.Timestamp,
        market_data: Optional[pd.DataFrame],
        strategy: str,
    ) -> Dict[str, float]:
        """Generate trading signals."""
        signals = {}
        
        with torch.no_grad():
            for symbol, df in symbol_data.items():
                # Find current index
                try:
                    idx = df.index.get_loc(date)
                except KeyError:
                    continue
                    
                if idx < self.extractor.lookback:
                    signals[symbol] = 0.0
                    continue
                    
                # Extract features
                try:
                    if market_data is not None:
                        market_aligned = market_data.reindex(df.index)
                        features = self.extractor(df, idx, market_aligned)
                    else:
                        features = self.extractor(df, idx)
                        
                    features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
                    
                    # Get model prediction
                    if hasattr(self.model, 'use_risk_head') and self.model.use_risk_head:
                        score, risk = self.model(features_tensor, return_risk=True)
                        score = score.cpu().item()
                        risk = risk.cpu().item()
                        
                        # Risk-adjusted signal
                        signal = score / (1 + risk)
                    else:
                        score = self.model(features_tensor).cpu().item()
                        signal = score
                        
                    signals[symbol] = signal
                    
                except Exception as e:
                    print(f"Error generating signal for {symbol} on {date}: {e}")
                    signals[symbol] = 0.0
                    
        return signals
    
    def _calculate_target_positions(
        self,
        signals: Dict[str, float],
        symbol_data: Dict[str, pd.DataFrame],
        date: pd.Timestamp,
        current_positions: Dict[str, float],
    ) -> Dict[str, float]:
        """Calculate target positions from signals."""
        target_positions = {}
        
        # Normalize signals
        signal_values = list(signals.values())
        if len(signal_values) == 0:
            return current_positions.copy()
            
        signal_mean = np.mean(signal_values)
        signal_std = np.std(signal_values)
        
        for symbol, signal in signals.items():
            # Z-score normalization
            if signal_std > 0:
                z_score = (signal - signal_mean) / signal_std
            else:
                z_score = 0
                
            # Convert to position
            if abs(z_score) < 0.5:
                position = 0.0
            else:
                position = np.tanh(z_score) * 0.5  # Max 50% position
                
            # Get recent returns for risk sizing
            df = symbol_data[symbol]
            idx = df.index.get_loc(date)
            if idx >= 20:
                recent_prices = df['Close'].iloc[idx-20:idx].values
                recent_returns = np.diff(recent_prices) / recent_prices[:-1]
            else:
                recent_returns = np.array([])
                
            # Apply risk management
            sized_position = self.risk_manager.calculate_position_size(
                symbol, position, recent_returns, current_positions
            )
            
            target_positions[symbol] = sized_position
            
        return target_positions
    
    def _calculate_position_value(
        self,
        positions: Dict[str, float],
        symbol_data: Dict[str, pd.DataFrame],
        date: pd.Timestamp
    ) -> float:
        """Calculate current value of positions."""
        value = 0.0
        
        for symbol, shares in positions.items():
            if shares == 0 or symbol not in symbol_data:
                continue
                
            df = symbol_data[symbol]
            if date in df.index:
                price = df.loc[date, 'Close']
                value += shares * price
                
        return value
    
    def _execute_trades(
        self,
        portfolio: Dict,
        target_positions: Dict[str, float],
        symbol_data: Dict[str, pd.DataFrame],
        date: pd.Timestamp,
    ) -> List[Dict]:
        """Execute trades to reach target positions."""
        trades = []
        total_value = portfolio['cash'] + self._calculate_position_value(
            portfolio['positions'], symbol_data, date
        )
        
        for symbol, target_weight in target_positions.items():
            if symbol not in symbol_data:
                continue
                
            df = symbol_data[symbol]
            if date not in df.index:
                continue
                
            price = df.loc[date, 'Close']
            current_shares = portfolio['positions'].get(symbol, 0)
            current_value = current_shares * price
            current_weight = current_value / total_value
            
            # Calculate trade
            weight_diff = target_weight - current_weight
            if abs(weight_diff) < 0.01:  # 1% threshold
                continue
                
            trade_value = weight_diff * total_value
            trade_shares = trade_value / price
            
            # Apply transaction costs
            cost = abs(trade_value) * (self.transaction_cost_bps / 10000)
            slippage = abs(trade_value) * (self.slippage_bps / 10000)
            
            # Execute trade
            portfolio['cash'] -= trade_value + cost + slippage
            portfolio['positions'][symbol] = current_shares + trade_shares
            
            trades.append({
                'date': date,
                'symbol': symbol,
                'shares': trade_shares,
                'price': price,
                'value': trade_value,
                'cost': cost + slippage,
            })
            
        return trades
    
    def _calculate_metrics(self, portfolio: Dict, initial_capital: float) -> Dict:
        """Calculate performance metrics."""
        returns = pd.Series(portfolio['values'], index=portfolio['dates'])
        daily_returns = returns.pct_change().dropna()
        
        # Basic metrics
        total_return = (returns.iloc[-1] - initial_capital) / initial_capital
        ann_return = (1 + total_return) ** (252 / len(returns)) - 1
        ann_vol = daily_returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        # Sortino ratio
        downside_returns = daily_returns[daily_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        sortino = ann_return / downside_vol if downside_vol > 0 else np.inf
        
        # Max drawdown
        peak = returns.expanding().max()
        drawdown = (returns - peak) / peak
        max_dd = drawdown.min()
        
        # Calmar ratio
        calmar = ann_return / abs(max_dd) if max_dd != 0 else np.inf
        
        # Win rate
        trades_df = pd.DataFrame(portfolio['trades'])
        if len(trades_df) > 0:
            trades_df['pnl'] = trades_df['value'] * -1  # negative value = buy
            win_rate = (trades_df['pnl'] > 0).mean()
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean()
            avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean()
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
            
        # Trading statistics
        n_trades = len(portfolio['trades'])
        if n_trades > 0:
            total_costs = sum(t['cost'] for t in portfolio['trades'])
            cost_impact = total_costs / initial_capital
        else:
            cost_impact = 0
            
        metrics = {
            'total_return': total_return,
            'annual_return': ann_return,
            'annual_volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'n_trades': n_trades,
            'cost_impact': cost_impact,
            'returns': daily_returns,
        }
        
        return metrics


def plot_results(results: Dict, save_path: Optional[Path] = None):
    """Plot backtest results."""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    portfolio = results['portfolio']
    returns = results['returns']
    
    # Equity curve
    ax = axes[0, 0]
    equity = pd.Series(portfolio['values'], index=portfolio['dates'])
    equity.plot(ax=ax, linewidth=2)
    ax.set_title('Equity Curve')
    ax.set_ylabel('Portfolio Value')
    ax.grid(True, alpha=0.3)
    
    # Drawdown
    ax = axes[0, 1]
    peak = equity.expanding().max()
    drawdown = (equity - peak) / peak * 100
    drawdown.plot(ax=ax, linewidth=2, color='red')
    ax.fill_between(drawdown.index, drawdown.values, alpha=0.3, color='red')
    ax.set_title('Drawdown')
    ax.set_ylabel('Drawdown %')
    ax.grid(True, alpha=0.3)
    
    # Returns distribution
    ax = axes[1, 0]
    returns.hist(bins=50, ax=ax, alpha=0.7, color='blue')
    ax.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.3%}')
    ax.set_title('Returns Distribution')
    ax.set_xlabel('Daily Return')
    ax.legend()
    
    # Rolling Sharpe
    ax = axes[1, 1]
    rolling_sharpe = returns.rolling(60).mean() / returns.rolling(60).std() * np.sqrt(252)
    rolling_sharpe.plot(ax=ax, linewidth=2)
    ax.axhline(results['sharpe_ratio'], color='red', linestyle='--', 
               label=f'Overall: {results["sharpe_ratio"]:.2f}')
    ax.set_title('Rolling Sharpe Ratio (60 days)')
    ax.set_ylabel('Sharpe Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Monthly returns heatmap
    ax = axes[2, 0]
    monthly_returns = returns.resample('M').sum()
    monthly_pivot = monthly_returns.groupby([monthly_returns.index.year, 
                                           monthly_returns.index.month]).mean().unstack()
    sns.heatmap(monthly_pivot, annot=True, fmt='.1%', cmap='RdYlGn', center=0, ax=ax)
    ax.set_title('Monthly Returns Heatmap')
    
    # Performance metrics
    ax = axes[2, 1]
    ax.axis('off')
    metrics_text = f"""
    Performance Metrics:
    
    Total Return: {results['total_return']:.2%}
    Annual Return: {results['annual_return']:.2%}
    Annual Volatility: {results['annual_volatility']:.2%}
    Sharpe Ratio: {results['sharpe_ratio']:.2f}
    Sortino Ratio: {results['sortino_ratio']:.2f}
    Max Drawdown: {results['max_drawdown']:.2%}
    Calmar Ratio: {results['calmar_ratio']:.2f}
    Win Rate: {results['win_rate']:.2%}
    Profit Factor: {results['profit_factor']:.2f}
    Number of Trades: {results['n_trades']}
    Cost Impact: {results['cost_impact']:.2%}
    """
    ax.text(0.1, 0.9, metrics_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True, help='Model checkpoint')
    parser.add_argument('--data-dir', type=Path, default='data/raw')
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'GOOGL', 'MSFT'])
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--initial-capital', type=float, default=1000000)
    parser.add_argument('--transaction-cost-bps', type=float, default=10.0)
    parser.add_argument('--strategy', type=str, default='threshold',
                        choices=['threshold', 'ranking', 'ml_signals'])
    parser.add_argument('--rebalance-freq', type=str, default='daily',
                        choices=['daily', 'weekly', 'monthly'])
    parser.add_argument('--output-dir', type=Path, default='backtest_results')
    parser.add_argument('--model-type', type=str, default='enhanced',
                        choices=['enhanced', 'lightweight'])
    args = parser.parse_args()
    
    # Setup device
    device = get_device()
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']
    
    # Create feature extractor
    extractor = AdvancedFeatureExtractor(
        lookback=config.get('lookback', 256),
        use_volume=True,
        use_technical=True,
        use_microstructure=True,
    )
    
    # Create model
    input_dim = 30  # This should match your feature dimension
    if args.model_type == 'enhanced':
        model = EnhancedDPOModel(
            input_dim=input_dim,
            d_model=config['model']['d_model'],
            n_heads=config['model']['n_heads'],
            n_layers=config['model']['n_layers'],
            d_ff=config['model']['d_ff'],
            use_risk_head=config.get('use_risk_head', True),
        )
    else:
        model = LightweightDPOModel(
            input_dim=input_dim,
            hidden_dim=config['model']['hidden_dim'],
            n_layers=config['model']['n_layers'],
        )
        
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Create risk manager
    risk_manager = RiskManager(
        max_position_size=0.2,
        max_leverage=1.0,
        vol_target=0.15,
        max_drawdown=0.2,
    )
    
    # Create backtester
    backtester = EnhancedBacktester(
        model=model,
        extractor=extractor,
        risk_manager=risk_manager,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    
    # Run backtest
    results = backtester.backtest(
        data_dir=args.data_dir,
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        rebalance_freq=args.rebalance_freq,
        strategy=args.strategy,
    )
    
    # Print results
    print("\nBacktest Results:")
    print("-" * 50)
    for metric, value in results.items():
        if metric not in ['portfolio', 'returns']:
            if isinstance(value, float):
                if 'return' in metric or 'ratio' in metric or 'rate' in metric:
                    print(f"{metric}: {value:.2%}")
                else:
                    print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
                
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results_df = pd.DataFrame({
        'date': results['portfolio']['dates'],
        'value': results['portfolio']['values'],
    })
    results_df.to_csv(args.output_dir / 'portfolio_values.csv', index=False)
    
    # Save trades
    if results['portfolio']['trades']:
        trades_df = pd.DataFrame(results['portfolio']['trades'])
        trades_df.to_csv(args.output_dir / 'trades.csv', index=False)
        
    # Plot results
    plot_results(results, args.output_dir / 'backtest_plot.png')


if __name__ == '__main__':
    main()