#!/usr/bin/env python
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger

from dpo_forecasting.models.dpo_model import DPOModel
from dpo_forecasting.data.dataset import PreferenceDataModule


@hydra.main(config_path="dpo_forecasting/configs", config_name="dpo", version_base=None)
def main(cfg: DictConfig) -> None:
    # DataModule
    dm = PreferenceDataModule(
        pairs_file=cfg.dataset.pairs_file,
        prices_dir=cfg.dataset.prices_dir,
        lookback=cfg.dataset.lookback,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        val_fraction=cfg.dataset.val_fraction,
    )

    # Model
    model = DPOModel(cfg)

    # Logger (W&B optional)
    logger = None
    if cfg.logger.use_wandb:
        logger = WandbLogger(
            project=cfg.logger.project,
            name=cfg.logger.run_name,
            log_model="all",
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # Trainer
    trainer = pl.Trainer(logger=logger, **cfg.trainer)
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()