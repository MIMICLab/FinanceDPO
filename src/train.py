#!/usr/bin/env python
"""Training entry point for DPO‑Finance with **TensorBoard** logging.

Hydra config `dpo.yaml` controls all hyper‑parameters.  Enable/disable the
TensorBoard logger via `logger.use_tb`.
"""
from __future__ import annotations

import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger

from dpo_forecasting.models.dpo_model import DPOModel
from dpo_forecasting.data.dataset import PreferenceDataModule



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
    # DataModule ---------------------------------------------------------
    dm = PreferenceDataModule(
        pairs_file=cfg.dataset.pairs_file,
        prices_dir=cfg.dataset.prices_dir,
        lookback=cfg.dataset.lookback,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        val_fraction=cfg.dataset.val_fraction,
    )
    dm.setup()  # <--- ensure num_features populated

    # Model --------------------------------------------------------------
    model = DPOModel(cfg)
    print(f"[INFO] model: {model.__class__.__name__}")

    # Logger -------------------------------------------------------------
    tb_logger = build_logger(cfg)

    # Trainer ------------------------------------------------------------
    trainer = pl.Trainer(logger=tb_logger, **cfg.trainer)
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
