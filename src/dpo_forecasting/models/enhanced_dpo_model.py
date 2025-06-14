# ─── src/dpo_forecasting/models/enhanced_dpo_model.py ─────────────────────
"""
Enhanced DPO model with advanced architectures and risk-aware objectives.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class AttentionPooling(nn.Module):
    """Attention-based pooling for sequence aggregation."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.attention = nn.Linear(d_model, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            pooled: [batch_size, d_model]
        """
        scores = self.attention(x)  # [batch_size, seq_len, 1]
        weights = F.softmax(scores, dim=1)
        pooled = (x * weights).sum(dim=1)
        return pooled


class TemporalConvBlock(nn.Module):
    """Temporal convolutional block for capturing local patterns."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, channels, seq_len]
        """
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out


class EnhancedDPOModel(nn.Module):
    """Enhanced DPO model with hybrid architecture.
    
    Features:
    - Multi-scale temporal convolutions
    - Transformer encoder with relative positional encoding
    - Attention pooling
    - Risk-aware output heads
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        conv_channels: List[int] = [128, 256, 512],
        conv_kernels: List[int] = [3, 5, 7],
        dropout: float = 0.1,
        max_seq_len: int = 512,
        use_risk_head: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.use_risk_head = use_risk_head
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Multi-scale temporal convolutions
        self.conv_blocks = nn.ModuleList()
        in_ch = d_model
        for out_ch, kernel in zip(conv_channels, conv_kernels):
            self.conv_blocks.append(
                TemporalConvBlock(in_ch, out_ch, kernel, dropout=dropout)
            )
            in_ch = out_ch
            
        # Project conv output to transformer dimension
        self.conv_projection = nn.Linear(conv_channels[-1], d_model)
        
        # Positional encoding
        self.pos_encoder = RelativePositionalEncoding(d_model, dropout, max_seq_len)
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = TransformerEncoder(encoder_layer, n_layers)
        
        # Attention pooling
        self.attention_pool = AttentionPooling(d_model)
        
        # Output heads
        self.preference_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        if use_risk_head:
            self.risk_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
                nn.Softplus()  # ensure positive risk
            )
            
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(
        self,
        x: torch.Tensor,
        return_risk: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, input_dim] or [batch_size, input_dim]
            return_risk: whether to return risk prediction
            
        Returns:
            preference_score: [batch_size, 1]
            risk_score: [batch_size, 1] (if return_risk=True)
        """
        # Handle both 2D and 3D inputs
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
            
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]
        
        # Convolutional feature extraction
        conv_input = x.transpose(1, 2)  # [batch_size, d_model, seq_len]
        for conv_block in self.conv_blocks:
            conv_input = conv_block(conv_input)
        conv_features = conv_input.transpose(1, 2)  # [batch_size, seq_len, conv_channels[-1]]
        conv_features = self.conv_projection(conv_features)  # [batch_size, seq_len, d_model]
        
        # Combine with original features
        x = x + conv_features
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer(x)  # [batch_size, seq_len, d_model]
        
        # Attention pooling
        pooled = self.attention_pool(x)  # [batch_size, d_model]
        
        # Preference prediction
        preference = self.preference_head(pooled)  # [batch_size, 1]
        
        if return_risk and self.use_risk_head:
            risk = self.risk_head(pooled)  # [batch_size, 1]
            return preference, risk
        
        return preference


class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding for transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create relative position embeddings
        self.relative_positions = nn.Parameter(
            torch.randn(2 * max_len - 1, d_model) * 0.02
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add relative positional encoding."""
        batch_size, seq_len, d_model = x.shape
        
        # For simplicity, use standard positional encoding here
        # In practice, implement proper relative attention
        position = torch.arange(seq_len, device=x.device).unsqueeze(0)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=x.device) * 
            -(math.log(10000.0) / d_model)
        )
        
        pe = torch.zeros(1, seq_len, d_model, device=x.device)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        x = x + pe
        return self.dropout(x)


class RiskAwareDPOLoss(nn.Module):
    """Risk-aware DPO loss function."""
    
    def __init__(
        self,
        beta: float = 0.1,
        risk_penalty: float = 0.01,
        margin: float = 0.0,
    ):
        super().__init__()
        self.beta = beta
        self.risk_penalty = risk_penalty
        self.margin = margin
        
    def forward(
        self,
        good_scores: torch.Tensor,
        bad_scores: torch.Tensor,
        good_risk: Optional[torch.Tensor] = None,
        bad_risk: Optional[torch.Tensor] = None,
        ref_good_scores: Optional[torch.Tensor] = None,
        ref_bad_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute risk-aware DPO loss.
        
        Args:
            good_scores: preference scores for good outcomes
            bad_scores: preference scores for bad outcomes
            good_risk: risk predictions for good outcomes
            bad_risk: risk predictions for bad outcomes
            ref_good_scores: reference model scores for KL regularization
            ref_bad_scores: reference model scores for KL regularization
        """
        # Basic DPO loss with margin
        logits = good_scores - bad_scores - self.margin
        dpo_loss = -F.logsigmoid(self.beta * logits).mean()
        
        # KL regularization
        if ref_good_scores is not None and ref_bad_scores is not None:
            ref_logits = ref_good_scores - ref_bad_scores
            kl_loss = F.kl_div(
                F.log_softmax(logits, dim=-1),
                F.softmax(ref_logits, dim=-1),
                reduction='batchmean'
            )
            dpo_loss = dpo_loss + 0.1 * kl_loss
            
        # Risk penalty
        if good_risk is not None and bad_risk is not None:
            # Penalize high risk for preferred outcomes
            risk_loss = self.risk_penalty * good_risk.mean()
            dpo_loss = dpo_loss + risk_loss
            
        return dpo_loss


class LightweightDPOModel(nn.Module):
    """Lightweight DPO model for faster inference.
    
    Uses GRU instead of Transformer for efficiency.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # GRU layers
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output dimension after GRU
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(gru_output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(gru_output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim] or [batch_size, input_dim]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        # Project input
        x = self.input_projection(x)
        
        # GRU encoding
        gru_out, _ = self.gru(x)  # [batch_size, seq_len, gru_output_dim]
        
        # Attention-based aggregation
        attn_scores = self.attention(gru_out)  # [batch_size, seq_len, 1]
        attn_weights = F.softmax(attn_scores, dim=1)
        weighted = gru_out * attn_weights
        aggregated = weighted.sum(dim=1)  # [batch_size, gru_output_dim]
        
        # Output
        output = self.output_head(aggregated)
        
        return output


__all__ = [
    "EnhancedDPOModel",
    "RiskAwareDPOLoss",
    "LightweightDPOModel",
]