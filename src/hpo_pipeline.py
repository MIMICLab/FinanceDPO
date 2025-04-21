#!/usr/bin/env python
"""
One‑shot pipeline for Optuna/Hydra sweeps:

1. Generate pairs + cache for the given lookback/lookahead
2. Call `train.py` on that cache
3. Exit with val_sharpe printed so Optuna can parse it
"""
from pathlib import Path
import subprocess, uuid, json, sys
import os

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="dpo_forecasting/configs", config_name="dpo")
def main(cfg: DictConfig):
    # unique tag for this Optuna/Hydra trial
    run_id = uuid.uuid4().hex[:8]

    # ensure child processes can import dpo_forecasting without editable install
    env = os.environ.copy()
    script_path = Path(__file__).resolve()
    # If this file is .../FinanceDPO/src/hpo_pipeline.py  → src_root = .../FinanceDPO/src
    # If it's .../FinanceDPO/src/dpo_forecasting/hpo_pipeline.py → src_root = .../FinanceDPO/src
    if script_path.parent.name == "src":
        src_root = script_path.parent
    else:
        # path looks like .../src/dpo_forecasting/hpo_pipeline.py
        src_root = script_path.parents[2]  # up to .../src
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(src_root), env.get("PYTHONPATH", "")]))
    print(f"[PIPELINE]  PYTHONPATH for subprocesses: {env['PYTHONPATH']}")

    # ---------------- hydra output dir (fallback to CWD) -------------
    try:
        out_dir = Path(cfg.hydra.runtime.output_dir)
    except Exception:
        out_dir = Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_file = out_dir / f"pairs_cache_{run_id}.pt"

    # ------------- 1. make_pairs ------------------------------------
    make_cmd = [
        sys.executable, "-m", "dpo_forecasting.data.make_pairs",
        "--prices-dir", cfg.dataset.prices_dir,
        "--lookahead", str(cfg.dataset.lookahead),
        "--lookback",  str(cfg.dataset.lookback),
        "--good-quantile", str(cfg.dataset.good_quantile),
        "--bad-quantile",  str(cfg.dataset.bad_quantile),
        "--cache-file", str(cache_file),
        "--skip-parquet",
    ]
    subprocess.check_call(make_cmd, env=env)

    # ------------- 2. train -----------------------------------------
    train_cmd = [
        sys.executable, "src/train.py",
        f"dataset.cache_file={cache_file}",
        f"trainer.max_epochs={cfg.trainer.max_epochs}",
        f"train.lr={cfg.train.lr}",
        f"train.kl_coeff={cfg.train.kl_coeff}",
    ]
    subprocess.check_call(train_cmd, env=env)

    # Lightning prints metrics as JSON; Optuna sweeper parses the first
    # line that is valid JSON with a `val_metric` key.
    # (train.py already logs val_sharpe with prog_bar=True)

if __name__ == "__main__":
    main()