"""Enhanced training script for DPO-based financial forecasting.

Features:
- Advanced model architectures
- Risk-aware training objectives
- Curriculum learning
- Dynamic batch sizing
- Comprehensive logging and checkpointing
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import yaml
from tqdm import tqdm

from dpo_forecasting.models.enhanced_dpo_model import (
    EnhancedDPOModel, RiskAwareDPOLoss, LightweightDPOModel
)
from dpo_forecasting.preprocessing.enhanced_dataset import (
    EnhancedPreferenceDataset, SequentialMarketDataset
)
from dpo_forecasting.preprocessing.advanced_extractors import (
    AdvancedFeatureExtractor, AdaptiveWindowExtractor
)
from dpo_forecasting.utils.device import get_device


class EnhancedTrainer:
    """Enhanced trainer with advanced training strategies."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device,
        output_dir: Path,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.output_dir = output_dir
        
        # Initialize optimizer
        self.optimizer = self._build_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._build_scheduler()
        
        # Loss function
        self.loss_fn = RiskAwareDPOLoss(
            beta=config.get('beta', 0.1),
            risk_penalty=config.get('risk_penalty', 0.01),
            margin=config.get('margin', 0.0),
        )
        
        # Mixed precision training
        self.scaler = GradScaler() if config.get('use_amp', True) else None
        
        # Logging
        self.writer = SummaryWriter(output_dir / 'tensorboard')
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Curriculum learning
        self.curriculum_stage = 0
        self.curriculum_thresholds = config.get('curriculum_thresholds', [0.7, 0.8, 0.9])
        
    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer with parameter groups."""
        # Different learning rates for different components
        param_groups = [
            {'params': self.model.input_projection.parameters(), 'lr': self.config['lr'] * 0.1},
            {'params': self.model.transformer.parameters(), 'lr': self.config['lr']},
            {'params': self.model.preference_head.parameters(), 'lr': self.config['lr'] * 2},
        ]
        
        if hasattr(self.model, 'risk_head'):
            param_groups.append({
                'params': self.model.risk_head.parameters(),
                'lr': self.config['lr'] * 2
            })
            
        optimizer_name = self.config.get('optimizer', 'adamw')
        if optimizer_name == 'adamw':
            return torch.optim.AdamW(
                param_groups,
                weight_decay=self.config.get('weight_decay', 0.01),
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(
                param_groups,
                momentum=0.9,
                weight_decay=self.config.get('weight_decay', 0.01),
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
            
    def _build_scheduler(self):
        """Build learning rate scheduler."""
        scheduler_name = self.config.get('scheduler', 'cosine')
        
        if scheduler_name == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs'],
                eta_min=self.config['lr'] * 0.01,
            )
        elif scheduler_name == 'onecycle':
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config['lr'],
                epochs=self.config['epochs'],
                steps_per_epoch=len(self.train_loader),
            )
        elif scheduler_name == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=self.config['lr'] * 0.001,
            )
        else:
            return None
            
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        accuracy_sum = 0.0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            good_features = batch['good_features'].to(self.device)
            bad_features = batch['bad_features'].to(self.device)
            
            # Curriculum learning: filter pairs based on difficulty
            if self.curriculum_stage < len(self.curriculum_thresholds):
                return_diff = (batch['good_return'] - batch['bad_return']).abs()
                threshold = self.curriculum_thresholds[self.curriculum_stage]
                mask = return_diff > threshold * return_diff.mean()
                
                if mask.sum() == 0:
                    continue
                    
                good_features = good_features[mask]
                bad_features = bad_features[mask]
            
            # Forward pass
            with autocast(enabled=self.scaler is not None):
                if hasattr(self.model, 'use_risk_head') and self.model.use_risk_head:
                    good_scores, good_risk = self.model(good_features, return_risk=True)
                    bad_scores, bad_risk = self.model(bad_features, return_risk=True)
                    
                    loss = self.loss_fn(
                        good_scores, bad_scores,
                        good_risk, bad_risk
                    )
                else:
                    good_scores = self.model(good_features)
                    bad_scores = self.model(bad_features)
                    
                    loss = self.loss_fn(good_scores, bad_scores)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            accuracy = ((good_scores > bad_scores).float().mean().item())
            accuracy_sum += accuracy
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{accuracy:.3f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Log to tensorboard
            global_step = epoch * len(self.train_loader) + batch_idx
            if batch_idx % 10 == 0:
                self.writer.add_scalar('train/loss', loss.item(), global_step)
                self.writer.add_scalar('train/accuracy', accuracy, global_step)
                
        # Scheduler step
        if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(total_loss / len(self.train_loader))
            else:
                self.scheduler.step()
                
        avg_loss = total_loss / len(self.train_loader)
        avg_accuracy = accuracy_sum / len(self.train_loader)
        
        # Update curriculum stage
        if avg_accuracy > 0.75 and self.curriculum_stage < len(self.curriculum_thresholds) - 1:
            self.curriculum_stage += 1
            print(f"Advanced to curriculum stage {self.curriculum_stage}")
            
        return avg_loss
    
    def validate(self, epoch: int) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        accuracy_sum = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                good_features = batch['good_features'].to(self.device)
                bad_features = batch['bad_features'].to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'use_risk_head') and self.model.use_risk_head:
                    good_scores, good_risk = self.model(good_features, return_risk=True)
                    bad_scores, bad_risk = self.model(bad_features, return_risk=True)
                    
                    loss = self.loss_fn(
                        good_scores, bad_scores,
                        good_risk, bad_risk
                    )
                else:
                    good_scores = self.model(good_features)
                    bad_scores = self.model(bad_features)
                    
                    loss = self.loss_fn(good_scores, bad_scores)
                
                total_loss += loss.item()
                accuracy = ((good_scores > bad_scores).float().mean().item())
                accuracy_sum += accuracy
                
        avg_loss = total_loss / len(self.val_loader)
        avg_accuracy = accuracy_sum / len(self.val_loader)
        
        # Log to tensorboard
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        self.writer.add_scalar('val/accuracy', avg_accuracy, epoch)
        
        return avg_loss, avg_accuracy
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'config': self.config,
            'curriculum_stage': self.curriculum_stage,
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.output_dir / 'latest.pt')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best.pt')
            
        # Save periodic checkpoint
        if epoch % 10 == 0:
            torch.save(checkpoint, self.output_dir / f'checkpoint_epoch_{epoch}.pt')
            
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.config['epochs']} epochs")
        
        for epoch in range(1, self.config['epochs'] + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_accuracy = self.validate(epoch)
            
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.3f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Early stopping
            if self.patience_counter >= self.config.get('patience', 20):
                print(f"Early stopping triggered after {epoch} epochs")
                break
                
        self.writer.close()
        print(f"Training completed. Best validation loss: {self.best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data-dir', type=Path, default='data/raw')
    parser.add_argument('--output-dir', type=Path, default='runs/enhanced')
    parser.add_argument('--model-type', type=str, default='enhanced',
                        choices=['enhanced', 'lightweight'])
    parser.add_argument('--resume', type=Path, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
        
    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(args.output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
        
    # Load symbols
    symbols = config.get('symbols', ['AAPL', 'GOOGL', 'MSFT'])
    
    # Create feature extractor
    if config.get('use_adaptive_window', False):
        extractor = AdaptiveWindowExtractor(
            min_lookback=config.get('min_lookback', 20),
            max_lookback=config.get('max_lookback', 252),
        )
    else:
        extractor = AdvancedFeatureExtractor(
            lookback=config.get('lookback', 256),
            use_volume=config.get('use_volume', True),
            use_technical=config.get('use_technical', True),
            use_microstructure=config.get('use_microstructure', True),
        )
    
    # Create dataset
    dataset = EnhancedPreferenceDataset(
        data_dir=args.data_dir,
        symbols=symbols,
        extractor=extractor,
        forward_days=config.get('forward_days', 8),
        preference_quantiles=tuple(config.get('preference_quantiles', [0.2, 0.8])),
        use_adaptive_thresholds=config.get('use_adaptive_thresholds', True),
        augment_data=config.get('augment_data', True),
        market_data_path=args.data_dir / 'SPY.csv' if (args.data_dir / 'SPY.csv').exists() else None,
    )
    
    # Split dataset
    val_size = int(len(dataset) * config.get('val_split', 0.1))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
    )
    
    # Get input dimension
    sample = dataset[0]
    input_dim = sample['good_features'].shape[-1]
    
    # Create model
    if args.model_type == 'enhanced':
        model = EnhancedDPOModel(
            input_dim=input_dim,
            d_model=config['model']['d_model'],
            n_heads=config['model']['n_heads'],
            n_layers=config['model']['n_layers'],
            d_ff=config['model']['d_ff'],
            dropout=config['model'].get('dropout', 0.1),
            use_risk_head=config.get('use_risk_head', True),
        )
    else:
        model = LightweightDPOModel(
            input_dim=input_dim,
            hidden_dim=config['model']['hidden_dim'],
            n_layers=config['model']['n_layers'],
            dropout=config['model'].get('dropout', 0.1),
        )
        
    model = model.to(device)
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Resumed from checkpoint: {args.resume}")
        
    # Create trainer
    trainer = EnhancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        output_dir=args.output_dir,
    )
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()