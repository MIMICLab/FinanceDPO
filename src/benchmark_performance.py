"""Benchmark script to compare original vs optimized implementations."""

import time
import gc
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class PerformanceBenchmark:
    """Benchmark suite for comparing implementations."""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}
        
    def benchmark_data_loading(
        self,
        pairs_file: str,
        prices_dir: str,
        n_epochs: int = 3,
        batch_size: int = 256,
    ) -> Dict[str, float]:
        """Benchmark data loading performance."""
        results = {}
        
        # Benchmark original dataset
        print("Benchmarking original dataset...")
        from dpo_forecasting.preprocessing.dataset import PreferenceDataModule
        
        orig_dm = PreferenceDataModule(
            pairs_file=pairs_file,
            prices_dir=prices_dir,
            batch_size=batch_size,
            num_workers=4,
        )
        orig_dm.setup()
        
        start_time = time.time()
        for epoch in range(n_epochs):
            for batch in tqdm(orig_dm.train_dataloader(), desc=f"Original epoch {epoch+1}"):
                pass
        orig_time = time.time() - start_time
        results["original_data_loading"] = orig_time / n_epochs
        
        # Clear memory
        del orig_dm
        gc.collect()
        torch.cuda.empty_cache()
        
        # Benchmark optimized dataset
        print("\nBenchmarking optimized dataset...")
        from dpo_forecasting.preprocessing.optimized_dataset import OptimizedDataModule
        
        opt_dm = OptimizedDataModule(
            pairs_file=pairs_file,
            prices_dir=prices_dir,
            batch_size=batch_size,
            num_workers=4,
            prefetch_factor=2,
        )
        opt_dm.setup()
        
        start_time = time.time()
        for epoch in range(n_epochs):
            for batch in tqdm(opt_dm.train_dataloader(), desc=f"Optimized epoch {epoch+1}"):
                pass
        opt_time = time.time() - start_time
        results["optimized_data_loading"] = opt_time / n_epochs
        
        results["data_loading_speedup"] = orig_time / opt_time
        
        return results
        
    def benchmark_model_forward(
        self,
        lookback: int = 256,
        batch_sizes: list = [32, 64, 128, 256],
        n_iterations: int = 100,
    ) -> Dict[str, float]:
        """Benchmark model forward pass performance."""
        results = {}
        
        # Model configuration
        cfg = {
            "model": {
                "d_model": 512,
                "n_heads": 8,
                "n_layers": 6,
                "ff_dim": 2048,
            },
            "train": {"lr": 1e-4},
        }
        
        for batch_size in batch_sizes:
            # Create dummy input
            dummy_input = torch.randn(batch_size, lookback - 1).to(self.device)
            
            # Benchmark original model
            from dpo_forecasting.models.dpo_model import DPOModel
            orig_model = DPOModel(cfg, lookback=lookback).to(self.device)
            orig_model.eval()
            
            # Warmup
            for _ in range(10):
                _ = orig_model(dummy_input)
                
            torch.cuda.synchronize()
            start_time = time.time()
            for _ in range(n_iterations):
                _ = orig_model(dummy_input)
            torch.cuda.synchronize()
            orig_time = time.time() - start_time
            
            # Clear memory
            del orig_model
            gc.collect()
            torch.cuda.empty_cache()
            
            # Benchmark optimized model
            from dpo_forecasting.models.optimized_dpo_model import OptimizedDPOModel
            opt_model = OptimizedDPOModel(cfg, lookback=lookback, use_flash=True).to(self.device)
            opt_model.eval()
            
            # Warmup
            for _ in range(10):
                _ = opt_model(dummy_input)
                
            torch.cuda.synchronize()
            start_time = time.time()
            for _ in range(n_iterations):
                _ = opt_model(dummy_input)
            torch.cuda.synchronize()
            opt_time = time.time() - start_time
            
            results[f"batch_{batch_size}_original"] = orig_time / n_iterations
            results[f"batch_{batch_size}_optimized"] = opt_time / n_iterations
            results[f"batch_{batch_size}_speedup"] = orig_time / opt_time
            
            # Clear memory
            del opt_model
            gc.collect()
            torch.cuda.empty_cache()
            
        return results
        
    def benchmark_memory_usage(
        self,
        lookback: int = 256,
        batch_size: int = 128,
    ) -> Dict[str, float]:
        """Benchmark memory usage."""
        results = {}
        
        cfg = {
            "model": {
                "d_model": 768,
                "n_heads": 12,
                "n_layers": 12,
                "ff_dim": 3072,
            },
            "train": {"lr": 1e-4},
        }
        
        # Original model memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        from dpo_forecasting.models.dpo_model import DPOModel
        orig_model = DPOModel(cfg, lookback=lookback).to(self.device)
        dummy_input = torch.randn(batch_size, lookback - 1).to(self.device)
        
        # Forward and backward
        loss = orig_model(dummy_input).sum()
        loss.backward()
        
        orig_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        results["original_memory_gb"] = orig_memory
        
        del orig_model, loss
        gc.collect()
        torch.cuda.empty_cache()
        
        # Optimized model memory (with gradient checkpointing)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        from dpo_forecasting.models.optimized_dpo_model import OptimizedDPOModel
        opt_model = OptimizedDPOModel(
            cfg, 
            lookback=lookback, 
            use_flash=True,
            use_checkpoint=True,
        ).to(self.device)
        
        # Forward and backward
        loss = opt_model(dummy_input).sum()
        loss.backward()
        
        opt_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        results["optimized_memory_gb"] = opt_memory
        results["memory_reduction"] = (orig_memory - opt_memory) / orig_memory
        
        return results
        
    def generate_report(self) -> str:
        """Generate performance comparison report."""
        report = []
        report.append("="*60)
        report.append("PERFORMANCE COMPARISON REPORT")
        report.append("="*60)
        
        if "data_loading" in self.results:
            report.append("\n## Data Loading Performance")
            data = self.results["data_loading"]
            report.append(f"Original: {data['original_data_loading']:.2f}s per epoch")
            report.append(f"Optimized: {data['optimized_data_loading']:.2f}s per epoch")
            report.append(f"Speedup: {data['data_loading_speedup']:.2f}x")
            
        if "model_forward" in self.results:
            report.append("\n## Model Forward Pass Performance")
            data = self.results["model_forward"]
            for batch_size in [32, 64, 128, 256]:
                if f"batch_{batch_size}_speedup" in data:
                    report.append(f"\nBatch size {batch_size}:")
                    report.append(f"  Original: {data[f'batch_{batch_size}_original']*1000:.2f}ms")
                    report.append(f"  Optimized: {data[f'batch_{batch_size}_optimized']*1000:.2f}ms")
                    report.append(f"  Speedup: {data[f'batch_{batch_size}_speedup']:.2f}x")
                    
        if "memory" in self.results:
            report.append("\n## Memory Usage")
            data = self.results["memory"]
            report.append(f"Original: {data['original_memory_gb']:.2f} GB")
            report.append(f"Optimized: {data['optimized_memory_gb']:.2f} GB")
            report.append(f"Reduction: {data['memory_reduction']*100:.1f}%")
            
        report.append("\n" + "="*60)
        return "\n".join(report)


def main():
    """Run performance benchmarks."""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running benchmarks on {device}")
    
    benchmark = PerformanceBenchmark(device)
    
    # Paths
    pairs_file = Path.home() / "Desktop/FinanceDPO/data/sp500_pairs.parquet"
    prices_dir = Path.home() / "Desktop/FinanceDPO/data/sp500_prices"
    
    # Run benchmarks
    print("\n1. Benchmarking data loading...")
    if pairs_file.exists() and prices_dir.exists():
        benchmark.results["data_loading"] = benchmark.benchmark_data_loading(
            str(pairs_file), str(prices_dir)
        )
    
    print("\n2. Benchmarking model forward pass...")
    benchmark.results["model_forward"] = benchmark.benchmark_model_forward()
    
    if device.type == "cuda":
        print("\n3. Benchmarking memory usage...")
        benchmark.results["memory"] = benchmark.benchmark_memory_usage()
    
    # Generate and print report
    report = benchmark.generate_report()
    print("\n" + report)
    
    # Save report
    with open("performance_report.txt", "w") as f:
        f.write(report)
    print("\nReport saved to performance_report.txt")


if __name__ == "__main__":
    main()