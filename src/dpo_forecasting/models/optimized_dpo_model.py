"""Optimized DPO Model with Flash Attention and performance improvements."""

from __future__ import annotations
from typing import Dict, Optional, Any, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# Try to import flash attention
try:
    from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    print("Flash Attention not available. Install with: pip install flash-attn")


class PrecomputedPositionalEncoding(nn.Module):
    """Precomputed positional encoding for efficiency."""
    
    def __init__(self, d_model: int, max_len: int = 5000, base: int = 10000):
        super().__init__()
        
        # Precompute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Optimized computation
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            -(math.log(base) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not parameter) for no gradients
        self.register_buffer("pe", pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.
        
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1)]


class FlashMultiheadAttention(nn.Module):
    """Multi-head attention using Flash Attention when available."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_flash: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_flash = use_flash and FLASH_ATTENTION_AVAILABLE
        
        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if not self.use_flash:
            self.scale = 1.0 / math.sqrt(self.d_k)
            
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with optional Flash Attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            attn_mask: Optional attention mask
            need_weights: Whether to return attention weights (not supported with Flash)
            
        Returns:
            output: Tensor of shape (batch_size, seq_len, d_model)
            attn_weights: Optional attention weights if need_weights=True
        """
        batch_size, seq_len, _ = x.shape
        
        # Combined QKV projection
        qkv = self.qkv_proj(x)  # (batch_size, seq_len, 3 * d_model)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, n_heads, seq_len, d_k)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if self.use_flash and not need_weights:
            # Use Flash Attention
            # Reshape for flash_attn format: (batch_size, seq_len, n_heads, d_k)
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            
            # Apply Flash Attention
            out = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout.p if self.training else 0.0,
                causal=False,
            )
            # out shape: (batch_size, seq_len, n_heads, d_k)
            
            # Reshape back
            out = out.reshape(batch_size, seq_len, self.d_model)
            attn_weights = None
            
        else:
            # Standard attention
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask == 0, -1e9)
                
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            out = torch.matmul(attn_weights, v)
            out = out.transpose(1, 2).contiguous()
            out = out.reshape(batch_size, seq_len, self.d_model)
            
            if not need_weights:
                attn_weights = None
                
        out = self.out_proj(out)
        return out, attn_weights


class OptimizedTransformerBlock(nn.Module):
    """Optimized transformer block with Flash Attention and fused operations."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_flash: bool = True,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        
        self.use_checkpoint = use_checkpoint
        
        # Self-attention
        self.self_attn = FlashMultiheadAttention(
            d_model, n_heads, dropout, use_flash
        )
        
        # Feed-forward network with fused operations
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with gradient checkpointing support."""
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, attn_mask, use_reentrant=False
            )
        return self._forward(x, attn_mask)
        
    def _forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, attn_mask)
        x = x + residual
        
        # Feed-forward with residual
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + residual
        
        return x


class OptimizedDPOModel(pl.LightningModule):
    """Optimized DPO model with performance improvements.
    
    Key optimizations:
    - Flash Attention for faster transformer blocks
    - Precomputed positional encodings
    - Gradient checkpointing for memory efficiency
    - Fused operations where possible
    - Optimized KL divergence computation
    """
    
    def __init__(
        self,
        cfg: Dict[str, Any],
        reference_net: Optional[nn.Module] = None,
        lookback: Optional[int] = None,
        use_flash: bool = True,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["reference_net"])
        self.cfg = cfg
        self.reference_net = reference_net
        
        # Model configuration
        L = lookback - 1 if lookback else 30
        d_model = cfg.model.get("d_model", 64)
        n_heads = cfg.model.get("n_heads", 4)
        ff_dim = cfg.model.get("ff_dim", 256)
        n_layers = cfg.model.get("n_layers", 2)
        dropout = cfg.model.get("dropout", 0.1)
        
        # Input embedding
        self.embed = nn.Linear(1, d_model)
        
        # Precomputed positional encoding
        self.pos_enc = PrecomputedPositionalEncoding(d_model, max_len=L + 100)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            OptimizedTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=ff_dim,
                dropout=dropout,
                use_flash=use_flash,
                use_checkpoint=use_checkpoint,
            )
            for _ in range(n_layers)
        ])
        
        # Output head with optimized pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Linear(d_model, 1)
        
        # Initialize weights
        self._init_weights()
        
        # Cache for reference model outputs
        self._ref_cache = {}
        self._ref_cache_size = 1000
        
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute preference score.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            
        Returns:
            Score tensor of shape (batch_size,)
        """
        # Embed and add positional encoding
        x = self.embed(x.unsqueeze(-1))  # (batch_size, seq_len, d_model)
        x = self.pos_enc(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Efficient pooling
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        x = self.pool(x).squeeze(-1)  # (batch_size, d_model)
        
        # Output score
        return self.fc_out(x).squeeze(-1)
        
    def compute_loss(
        self,
        good: torch.Tensor,
        bad: torch.Tensor,
        label: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute DPO loss with optional KL regularization.
        
        Returns:
            loss: Total loss
            metrics: Dictionary of metrics for logging
        """
        # Compute scores
        s_good = self.score(good)
        s_bad = self.score(bad)
        
        # DPO loss
        if label is not None:
            pref = label.float()
            delta = torch.where(pref > 0.5, s_good - s_bad, s_bad - s_good)
            dpo_loss = -F.logsigmoid(delta).mean()
        else:
            dpo_loss = -F.logsigmoid(s_good - s_bad).mean()
            
        metrics = {"dpo_loss": dpo_loss}
        total_loss = dpo_loss
        
        # KL regularization with caching
        kl_coeff = self.cfg.train.get("kl_coeff", 0.0)
        if self.reference_net is not None and kl_coeff > 0:
            # Compute reference scores with caching
            with torch.no_grad():
                # Create cache keys
                good_key = good.data_ptr()
                bad_key = bad.data_ptr()
                
                # Check cache
                if good_key in self._ref_cache:
                    ref_good = self._ref_cache[good_key]
                else:
                    ref_good = self.reference_net.score(good)
                    self._update_cache(good_key, ref_good)
                    
                if bad_key in self._ref_cache:
                    ref_bad = self._ref_cache[bad_key]
                else:
                    ref_bad = self.reference_net.score(bad)
                    self._update_cache(bad_key, ref_bad)
                    
            # KL divergence as MSE of score differences
            kl = F.mse_loss(s_good - s_bad, ref_good - ref_bad)
            total_loss = total_loss + kl_coeff * kl
            metrics["kl"] = kl
            
        return total_loss, metrics
        
    def _update_cache(self, key: int, value: torch.Tensor):
        """Update reference cache with LRU eviction."""
        if len(self._ref_cache) >= self._ref_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._ref_cache))
            del self._ref_cache[oldest_key]
        self._ref_cache[key] = value.detach()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for inference."""
        return self.score(x)
        
    def training_step(self, batch, batch_idx):
        """Training step."""
        if isinstance(batch, (list, tuple)):
            good, bad, label = batch[0], batch[1], batch[2] if len(batch) > 2 else None
        else:
            good = batch["good"]
            bad = batch["bad"]
            label = batch.get("label")
            
        loss, metrics = self.compute_loss(good, bad, label)
        
        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        for name, value in metrics.items():
            self.log(f"train_{name}", value)
            
        return loss
        
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        if isinstance(batch, (list, tuple)):
            good, bad, label = batch[0], batch[1], batch[2] if len(batch) > 2 else None
        else:
            good = batch["good"]
            bad = batch["bad"]
            label = batch.get("label")
            
        loss, metrics = self.compute_loss(good, bad, label)
        
        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        for name, value in metrics.items():
            self.log(f"val_{name}", value)
            
    def configure_optimizers(self):
        """Configure optimizers with optional weight decay."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.named_parameters():
            if "bias" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
                
        weight_decay = self.cfg.train.get("weight_decay", 0.01)
        
        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.cfg.train.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # Optional learning rate scheduler
        use_scheduler = self.cfg.train.get("use_scheduler", False)
        if use_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.cfg.train.get("max_epochs", 100),
                eta_min=self.cfg.train.lr * 0.1,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
            
        return optimizer