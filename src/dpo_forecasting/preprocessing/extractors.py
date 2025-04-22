# ─── src/dpo_forecasting/data/extractors.py ─────────────────────────────
"""
Light‑weight feature extractor utilities for FinanceDPO.

Currently implemented
---------------------
ReturnWindowExtractor
    Returns a trailing vector of simple percentage returns for a fixed
    look‑back horizon.  Kept minimal so it can be imported safely by
    PyTorch DataLoader worker processes.
"""

from __future__ import annotations
from typing import Any
import numpy as np


class ReturnWindowExtractor:
    """Extract a rolling window of simple returns.

    Parameters
    ----------
    lookback : int
        Number of closing prices to include in the window (≥ 2).  The
        resulting feature vector has length ``lookback − 1`` consisting
        of simple returns ``(p[t] − p[t‑1]) / p[t‑1]``.
    """

    def __init__(self, lookback: int):
        if lookback < 2:
            raise ValueError("lookback must be ≥ 2 (need at least two prices)")
        self.lookback = int(lookback)

    def __call__(self, close: np.ndarray, idx: int) -> np.ndarray:
        """Compute the return window ending just before index *idx*.
        
        Returns
        -------
        numpy.ndarray
            1‑D float32 array of length ``lookback − 1`` containing
            simple percentage returns for the specified window.
        """
        if idx < self.lookback:
            raise IndexError("idx must be ≥ lookback")
        window = close[idx - self.lookback : idx].astype(np.float32)
        pct = (window[1:] - window[:-1]) / window[:-1]
        return pct

    # Enable safe pickling for PyTorch DataLoader workers
    def __getstate__(self) -> Any:
        return {"lookback": self.lookback}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)


__all__ = ["ReturnWindowExtractor"]