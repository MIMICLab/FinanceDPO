#!/usr/bin/env python
"""Training entry point for DPO‑Finance with **TensorBoard** logging.

Hydra config `dpo.yaml` controls all hyper‑parameters.  Enable/disable the
TensorBoard logger via `logger.use_tb`.
"""
from __future__ import annotations

import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from pathlib import Path

from dpo_forecasting.models.dpo_model import DPOModel
from dpo_forecasting.data.dataset import PreferenceDataModule
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


@hydra.main(config_path="dpo_forecasting/configs", config_name="dpo", version_base=None)
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

    # DataModule ---------------------------------------------------------
    dm = PreferenceDataModule(
        pairs_file=pairs_path,
        cache_file=cache_path,
        prices_dir=cfg.dataset.prices_dir,
        lookback=inferred_lookback,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        val_fraction=cfg.dataset.val_fraction,
    )
    dm.setup()  # <--- ensure num_features populated

    # Model --------------------------------------------------------------
    model = DPOModel(cfg, lookback=inferred_lookback)
    print(f"[INFO] model: {model.__class__.__name__}")

    # Logger -------------------------------------------------------------
    tb_logger = build_logger(cfg)

    # Trainer ------------------------------------------------------------
    trainer = pl.Trainer(logger=tb_logger, **cfg.trainer)
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
