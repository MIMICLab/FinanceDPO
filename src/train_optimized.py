"""Optimized training script with all performance improvements."""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.strategies import DDPStrategy
import hydra
from omegaconf import DictConfig, OmegaConf


class OptimizedTrainer:
    """Trainer with optimized settings for DPO models."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories."""
        self.output_dir = Path(self.cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def create_callbacks(self) -> list:
        """Create optimized callbacks."""
        callbacks = []
        
        # Model checkpoint with optimization
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            filename="{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            every_n_epochs=1,
            save_on_train_epoch_end=False,  # Save at validation time
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        if self.cfg.train.get("early_stopping", True):
            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=self.cfg.train.get("patience", 10),
                mode="min",
                min_delta=1e-4,
            )
            callbacks.append(early_stop)
            
        # Learning rate monitor
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))
        
        # Rich progress bar
        callbacks.append(RichProgressBar())
        
        return callbacks
        
    def create_trainer(self) -> pl.Trainer:
        """Create optimized PyTorch Lightning trainer."""
        # Determine strategy
        if torch.cuda.device_count() > 1:
            strategy = DDPStrategy(
                find_unused_parameters=False,
                gradient_as_bucket_view=True,  # Optimization for DDP
            )
        else:
            strategy = "auto"
            
        # Create trainer with optimizations
        trainer = pl.Trainer(
            max_epochs=self.cfg.train.epochs,
            accelerator="auto",
            devices="auto",
            strategy=strategy,
            precision=self.cfg.train.get("precision", "16-mixed"),
            callbacks=self.create_callbacks(),
            gradient_clip_val=self.cfg.train.get("gradient_clip", 1.0),
            accumulate_grad_batches=self.cfg.train.get("accumulate_grad_batches", 1),
            log_every_n_steps=10,
            val_check_interval=self.cfg.train.get("val_check_interval", 1.0),
            num_sanity_val_steps=2,
            benchmark=True,  # Enable CUDNN benchmark for fixed input sizes
            deterministic=False,  # Allow non-deterministic ops for speed
            profiler=self.cfg.get("profiler", None),
            enable_model_summary=True,
            default_root_dir=self.output_dir,
        )
        
        return trainer
        
    def load_data_module(self) -> pl.LightningDataModule:
        """Load optimized data module."""
        # Use optimized dataset if specified
        if self.cfg.data.get("use_optimized", True):
            from dpo_forecasting.preprocessing.optimized_dataset import OptimizedDataModule
            
            data_module = OptimizedDataModule(
                pairs_file=self.cfg.data.pairs_file,
                prices_dir=self.cfg.data.prices_dir,
                lookback=self.cfg.data.lookback,
                cache_file=self.cfg.data.get("cache_file"),
                batch_size=self.cfg.train.batch_size,
                num_workers=self.cfg.data.get("num_workers", 4),
                val_fraction=self.cfg.data.get("val_fraction", 0.1),
                prefetch_factor=self.cfg.data.get("prefetch_factor", 2),
                persistent_workers=True,
            )
        else:
            from dpo_forecasting.preprocessing.dataset import PreferenceDataModule
            
            data_module = PreferenceDataModule(
                pairs_file=self.cfg.data.pairs_file,
                prices_dir=self.cfg.data.prices_dir,
                lookback=self.cfg.data.lookback,
                cache_file=self.cfg.data.get("cache_file"),
                batch_size=self.cfg.train.batch_size,
                num_workers=self.cfg.data.get("num_workers", 4),
                val_fraction=self.cfg.data.get("val_fraction", 0.1),
            )
            
        return data_module
        
    def create_model(
        self,
        data_module: pl.LightningDataModule,
        reference_net: Optional[torch.nn.Module] = None,
    ) -> pl.LightningModule:
        """Create optimized model."""
        # Use optimized model if specified
        if self.cfg.model.get("use_optimized", True):
            from dpo_forecasting.models.optimized_dpo_model import OptimizedDPOModel
            
            model = OptimizedDPOModel(
                cfg=self.cfg,
                reference_net=reference_net,
                lookback=self.cfg.data.lookback,
                use_flash=self.cfg.model.get("use_flash", True),
                use_checkpoint=self.cfg.model.get("use_checkpoint", False),
            )
        else:
            from dpo_forecasting.models.dpo_model import DPOModel
            
            model = DPOModel(
                cfg=self.cfg,
                reference_net=reference_net,
                lookback=self.cfg.data.lookback,
            )
            
        return model
        
    def load_reference_model(self) -> Optional[torch.nn.Module]:
        """Load reference model for KL regularization."""
        if not self.cfg.train.get("kl_coeff", 0) > 0:
            return None
            
        ref_checkpoint = self.cfg.train.get("reference_checkpoint")
        if not ref_checkpoint:
            return None
            
        print(f"Loading reference model from {ref_checkpoint}")
        checkpoint = torch.load(ref_checkpoint, map_location="cpu")
        
        # Create reference model
        ref_cfg = checkpoint["hyper_parameters"]["cfg"]
        if self.cfg.model.get("use_optimized", True):
            from dpo_forecasting.models.optimized_dpo_model import OptimizedDPOModel
            reference_net = OptimizedDPOModel(
                cfg=ref_cfg,
                lookback=self.cfg.data.lookback,
            )
        else:
            from dpo_forecasting.models.dpo_model import DPOModel
            reference_net = DPOModel(
                cfg=ref_cfg,
                lookback=self.cfg.data.lookback,
            )
            
        # Load weights
        state_dict = checkpoint["state_dict"]
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        reference_net.load_state_dict(state_dict)
        
        # Freeze reference model
        for param in reference_net.parameters():
            param.requires_grad = False
            
        reference_net.eval()
        return reference_net
        
    def run(self):
        """Run optimized training."""
        # Setup data
        print("Setting up data module...")
        data_module = self.load_data_module()
        data_module.setup()
        
        # Load reference model if needed
        reference_net = self.load_reference_model()
        
        # Create model
        print("Creating model...")
        model = self.create_model(data_module, reference_net)
        
        # Create trainer
        print("Creating trainer...")
        trainer = self.create_trainer()
        
        # Compile model for additional speedup (PyTorch 2.0+)
        if hasattr(torch, "compile") and self.cfg.get("compile_model", True):
            print("Compiling model with torch.compile()...")
            model = torch.compile(
                model,
                mode=self.cfg.get("compile_mode", "default"),
                fullgraph=False,
            )
        
        # Auto-scale batch size if requested
        if self.cfg.train.get("auto_scale_batch_size", False):
            print("Finding optimal batch size...")
            trainer.tuner.scale_batch_size(
                model,
                datamodule=data_module,
                mode="power",
                steps_per_trial=3,
                init_val=self.cfg.train.batch_size,
            )
            
        # Find learning rate if requested
        if self.cfg.train.get("auto_lr", False):
            print("Finding optimal learning rate...")
            lr_finder = trainer.tuner.lr_find(
                model,
                datamodule=data_module,
                min_lr=1e-6,
                max_lr=1e-2,
                num_training=100,
            )
            suggested_lr = lr_finder.suggestion()
            print(f"Suggested learning rate: {suggested_lr}")
            model.hparams.cfg.train.lr = suggested_lr
            
        # Save configuration
        config_path = self.output_dir / "config.yaml"
        with open(config_path, "w") as f:
            OmegaConf.save(self.cfg, f)
            
        # Train model
        print("Starting training...")
        trainer.fit(model, datamodule=data_module)
        
        # Save final metrics
        if trainer.callback_metrics:
            metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
            metrics_path = self.output_dir / "final_metrics.yaml"
            with open(metrics_path, "w") as f:
                OmegaConf.save(metrics, f)
                
        print(f"\nTraining completed! Results saved to {self.output_dir}")


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    # Print configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Run training
    trainer = OptimizedTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()