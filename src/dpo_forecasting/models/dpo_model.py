"""PyTorch Lightning implementation of a Direct Preference Optimization model.

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

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


# ───────────────────────────────────────────────────────── helper nets ──


def mlp(input_dim: int, hidden: Sequence[int]) -> nn.Sequential:
    layers = []
    prev = input_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, 1))  # final scalar score
    return nn.Sequential(*layers)


# ─────────────────────────────────────────────────────────────  model ──


class DPOModel(pl.LightningModule):
    """Direct Preference Optimization model.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Hydra/OMEGACONF configuration tree.  The following keys are expected::

            cfg.model.input_dim      # int
            cfg.model.hidden_sizes   # list[int]
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

        inp_dim: int = cfg.model.input_dim
        hidden: Sequence[int] = cfg.model.hidden_sizes
        self.net = mlp(inp_dim, hidden)

    # ───────────────────────────────────────────────────── forward / loss ──

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Return a scalar preference score for each input *x* (shape [N])."""
        return self.net(x).squeeze(-1)

    def dpo_loss(self, s_good: torch.Tensor, s_bad: torch.Tensor) -> torch.Tensor:
        # pairwise logistic loss  −log σ(Δ)
        return -F.logsigmoid(s_good - s_bad).mean()

    def step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        good, bad = batch["good"], batch["bad"]
        s_good = self.score(good)
        s_bad = self.score(bad)
        loss = self.dpo_loss(s_good, s_bad)

        if self.reference_net is not None and self.cfg.train.get("kl_coeff", 0.0) > 0:
            with torch.no_grad():
                ref_good = self.reference_net.score(good)
                ref_bad = self.reference_net.score(bad)
            # KL term approximated as MSE between logits (simplified)
            kl = F.mse_loss(s_good - s_bad, ref_good - ref_bad)
            loss = loss + self.cfg.train.kl_coeff * kl
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
