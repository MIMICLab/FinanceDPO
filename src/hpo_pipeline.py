#!/usr/bin/env python
"""
One‑shot pipeline for Optuna/Hydra sweeps:

1. Generate pairs + cache for the given lookback/lookahead
2. Call `train.py` on that cache
3. Exit with val_sharpe printed so Optuna can parse it
"""
from pathlib import Path
import subprocess, uuid, json, sys

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="dpo_forecasting/configs", config_name="dpo")
def main(cfg: DictConfig):
    # ------------- unique paths per trial ---------------------------
    run_id = uuid.uuid4().hex[:8]
    cache_file = Path(cfg.runtime.output_dir) / f"pairs_cache_{run_id}.pt"

    # ------------- 1. make_pairs ------------------------------------
    make_cmd = [
        sys.executable, "-m", "dpo_forecasting.data.make_pairs",
        "--prices-dir", cfg.dataset.prices_dir,
        "--lookahead", str(cfg.lookahead),
        "--lookback", str(cfg.dataset.lookback),
        "--good-quantile", str(cfg.good_quantile),
        "--bad-quantile" , str(cfg.bad_quantile),
        "--cache-file", str(cache_file),
        "--skip-parquet",
    ]
    subprocess.check_call(make_cmd)

    # ------------- 2. train -----------------------------------------
    train_cmd = [
        sys.executable, "src/train.py",
        f"+dataset.cache_file={cache_file}",
        f"+trainer.max_epochs={cfg.trainer.max_epochs}",
        f"+train.lr={cfg.train.lr}",
        f"+train.kl_coeff={cfg.train.kl_coeff}",
    ]
    subprocess.check_call(train_cmd)

    # Lightning prints metrics as JSON; Optuna sweeper parses the first
    # line that is valid JSON with a `val_metric` key.
    # (train.py already logs val_sharpe with prog_bar=True)

if __name__ == "__main__":
    main()