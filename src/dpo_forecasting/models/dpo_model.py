"""PyTorch Lightning implementation of a Direct Preference Optimization model.

The module expects batches in the following dictionary format::

    {
        "good": Tensor[N, C, ...],   # preferred contexts
        "bad":  Tensor[N, C, ...]    # dispreferred contexts
    }

where *N* is batch size and the tensors are identical in shape.  The forward
network produces a scalar **score** for each context.  The loss is the
log‑sigmoid pairwise objective from the original DPO paper::

    L = -E[  log σ(f(x_good) - f(x_bad))  ]

Optionally, if a reference model is provided ("f_ref"), a KL divergence term is
added to keep the fine‑tuned model close to the reference distribution.
"""
from __future__ import annotations

from typing import Sequence, Optional, Dict, Any

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


# ──────────────────────────────────────────── positional enc ──
class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # shape (1, L, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (N, L, d_model)
        return x + self.pe[:, : x.size(1)]


# ─────────────────────────────────────────────────────────────  model ──


class DPOModel(pl.LightningModule):
    """Direct Preference Optimization model.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Hydra/OMEGACONF configuration tree.  The following keys are expected::

            cfg.model.d_model         # model dimension (default 64)
            cfg.model.n_heads         # attention heads
            cfg.model.ff_dim          # feed‑forward dimension
            cfg.model.n_layers        # encoder layers
            cfg.train.lr             # float learning rate
            cfg.train.kl_coeff       # float (optional) – weight for KL term
    reference_net : Optional[nn.Module]
        Frozen model p_ref(x) for KL regularization.  If provided, its outputs
        should be comparable (same scaling) to the main network.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        reference_net: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["reference_net"])
        self.cfg = cfg
        self.reference_net = reference_net

        L = cfg.dataset.lookback - 1  # sequence length
        d_model: int = cfg.model.get("d_model", 64)
        n_heads: int = cfg.model.get("n_heads", 4)
        ff_dim: int = cfg.model.get("ff_dim", 256)
        n_layers: int = cfg.model.get("n_layers", 2)

        self.embed = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=L)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(d_model, 1)

    # ───────────────────────────────────────────────────── forward / loss ──

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute preference score with a Transformer encoder.

        Parameters
        ----------
        x : torch.Tensor
            Shape (N, L) where L = lookback - 1.
        """
        # (N, L) -> (N, L, 1)
        z = self.embed(x.unsqueeze(-1))
        z = self.pos_enc(z)
        z = self.encoder(z)           # (N, L, d_model)
        z = z.mean(dim=1)             # simple average pooling
        return self.fc_out(z).squeeze(-1)

    def dpo_loss(self, s_good: torch.Tensor, s_bad: torch.Tensor) -> torch.Tensor:
        # pairwise logistic loss  −log σ(Δ)
        return -F.logsigmoid(s_good - s_bad).mean()

    def step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Accept either dict(bad=..., good=...) or tuple/list (good, bad [, label])
        if isinstance(batch, (list, tuple)):
            good, bad, *rest = batch
            label = rest[0] if rest else None
        else:
            good = (
                batch.get("good")
                or batch.get("ctx_good")
                or batch.get("x_good")
                or next((v for k, v in batch.items() if k.endswith("_good")), None)
            )
            bad = (
                batch.get("bad")
                or batch.get("ctx_bad")
                or batch.get("x_bad")
                or next((v for k, v in batch.items() if k.endswith("_bad")), None)
            )
            label = batch.get("label")

        if good is None or bad is None:
            raise KeyError(
                "Batch must provide `good` and `bad` tensors "
                "either as dict keys or as a (good, bad, …) tuple."
            )

        s_good = self.score(good)
        s_bad = self.score(bad)

        # ── pairwise loss ────────────────────────────────────────────────
        if label is not None:
            pref = label.float()
            # label==1  → good preferred; 0 → bad preferred
            delta = torch.where(pref > 0.5, s_good - s_bad, s_bad - s_good)
            loss = -F.logsigmoid(delta).mean()
        else:
            loss = self.dpo_loss(s_good, s_bad)

        # ── optional KL regularisation ───────────────────────────────────
        kl_coeff = self.cfg.train.get("kl_coeff", 0.0)
        if self.reference_net is not None and kl_coeff > 0:
            with torch.no_grad():
                ref_good = self.reference_net.score(good)
                ref_bad = self.reference_net.score(bad)
            kl = F.mse_loss(s_good - s_bad, ref_good - ref_bad)
            loss = loss + kl_coeff * kl
            self.log("kl", kl, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    # ──────────────────────────────────────────────────────  pytorch‑lit ──

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # for inference
        return self.score(x)

    def training_step(self, batch: Dict[str, torch.Tensor], _: int):
        loss = self.step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], _: int):
        loss = self.step(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.train.lr)
