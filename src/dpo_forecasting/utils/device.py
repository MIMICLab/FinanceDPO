"""
Device helper

Returns **cuda → mps → cpu** in that order, so one call works
everywhere (NVIDIA GPU, Apple Silicon, pure CPU).
"""
from __future__ import annotations
import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")