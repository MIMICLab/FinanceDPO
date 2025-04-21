#!/usr/bin/env python
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig
from dpo_forecasting.models.dpo_model import DPOModel
from dpo_forecasting.data.dataset import PreferenceDataModule

@hydra.main(config_path=None)
def main(cfg: DictConfig):
    dm = PreferenceDataModule('data/pairs.parquet', 'data/raw')
    model = DPOModel({'model': {'input_dim': 29, 'hidden_sizes': [128,128]}, 'train': {'lr':1e-3}})
    trainer = pl.Trainer(max_epochs=1, logger=False, enable_checkpointing=False)
    trainer.fit(model, datamodule=dm)

if __name__ == '__main__':
    main()
