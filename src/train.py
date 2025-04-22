#!/usr/bin/env python
"""Training entry point for DPO‑Finance with **TensorBoard** logging.

Hydra config `dpo.yaml` controls all hyper‑parameters.  Enable/disable the
TensorBoard logger via `logger.use_tb`.
"""
from __future__ import annotations

import pytorch_lightning as pl
from pytorch_lightning.tuner import Tuner
from omegaconf import DictConfig
from omegaconf import OmegaConf
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from pathlib import Path

from dpo_forecasting.models.dpo_model import DPOModel
from dpo_forecasting.preprocessing.dataset import PreferenceDataModule

import pandas as pd
import numpy as np

def build_logger(cfg: DictConfig):
    if not cfg.logger.use_tb:
        return None
    return TensorBoardLogger(
        save_dir="tb_logs",
        name=cfg.logger.experiment_name,
        default_hp_metric=False,
    )



def main(cfg: DictConfig) -> None:
    # ------------------------------------------------------------
    # Infer look‑back automatically, preferring Torch cache if provided.
    if cfg.dataset.get("cache_file") and Path(cfg.dataset.cache_file).exists():
        obj = torch.load(cfg.dataset.cache_file, map_location="cpu")
        inferred_lookback = obj["feat_good"].shape[1] + 1
        print(f"[INFO] using lookback = {inferred_lookback} (from cache)")
        cache_path = cfg.dataset.cache_file
        pairs_path = None
    else:
        # Fall back to reading the Parquet file for feat length
        sample_feat_len = (
            pd.read_parquet(cfg.dataset.pairs_file, columns=["feat_good"])
            ["feat_good"]
            .iloc[0]
        )
        inferred_lookback = (
            len(sample_feat_len) + 1
            if isinstance(sample_feat_len, (list, tuple, np.ndarray))
            else cfg.dataset.lookback
        )

        print(f"[INFO] using lookback = {inferred_lookback} (from parquet)")
        cache_path = None
        pairs_path = cfg.dataset.pairs_file
    # ------------------------------------------------------------
    # -----------------------------------------------------------------
    # Build an initial DataModule with a small safe batch (32) so
    # Lightning tuner can probe GPU memory.
    tmp_bs = 32 if cfg.dataset.batch_size == "auto" else cfg.dataset.batch_size
    dm = PreferenceDataModule(
        pairs_file=pairs_path,
        cache_file=cache_path,
        prices_dir=cfg.dataset.prices_dir,
        lookback=inferred_lookback,
        batch_size=tmp_bs,
        num_workers=cfg.dataset.num_workers,
        val_fraction=cfg.dataset.val_fraction,
    )
    dm.setup()

    # Model --------------------------------------------------------------
    model = DPOModel(cfg, lookback=inferred_lookback)
    if cfg.model.get("reference_net") is not None:
        # Load reference model from checkpoint
        if not Path(cfg.model.reference_net).exists():
            raise FileNotFoundError(
                f"Reference model checkpoint not found: {cfg.model.reference_net}"
            )
        # Reference model for KL regularization
        ref_model = DPOModel(cfg, lookback=inferred_lookback)
        ref_model.load_state_dict(
            torch.load(cfg.model.reference_net, map_location="cpu")
        )
        model.reference_net = ref_model
        model.reference_net.eval()
        model.reference_net.requires_grad_(False)
        print(f"[INFO] reference model loaded from {cfg.model.reference_net}") 
    else:
        model.reference_net = None
        print("[INFO] no reference model provided")

    # Logger -------------------------------------------------------------
    tb_logger = build_logger(cfg)

    # Trainer ------------------------------------------------------------
    trainer = pl.Trainer(
        logger=tb_logger,
        **cfg.trainer,
    )

    if cfg.dataset.batch_size == "auto":
        print("[INFO] Auto‑scaling batch size …")              
        tuner = Tuner(trainer)
        new_bs = tuner.scale_batch_size(
            model,
            datamodule=dm,
            mode="power",         
            init_val=64,         
        )
        cfg.dataset.batch_size = int(new_bs)
        print(f"[INFO] batch_size tuned → {new_bs}")

        # rebuild DataModule with the tuned batch size
        dm = PreferenceDataModule(
            pairs_file=pairs_path,
            cache_file=cache_path,
            prices_dir=cfg.dataset.prices_dir,
            lookback=inferred_lookback,
            batch_size=new_bs,
            num_workers=cfg.dataset.num_workers,
            val_fraction=cfg.dataset.val_fraction,
        )
        dm.setup()

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Finance‑DPO model")
    parser.add_argument(
        "--config",
        "-c",
        default="src/dpo_forecasting/configs/findpo_base.yaml",
        help="Path to a YAML config file",
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    main(cfg)
