"""Unit tests for core DPO‑Finance components."""
import math
import numpy as np
import torch

from dpo_forecasting.models.dpo_model import DPOModel
from dpo_forecasting.preprocessing.dataset import ReturnWindowExtractor
from dpo_forecasting.preprocessing.dataset import PreferencePairDataset


def test_dpo_pairwise_loss():
    """Check that DPO loss equals manual −log σ(Δ) average."""
    cfg = {
        "model": {"input_dim": 4, "hidden_sizes": [8]},
        "train": {"lr": 1e-3},
    }
    model = DPOModel(cfg)
    # Fixed weights for deterministic behaviour
    torch.nn.init.constant_(model.net[0].weight, 0.1)
    torch.nn.init.constant_(model.net[0].bias, 0.0)

    good = torch.tensor([[1.0, 0.0, -1.0, 0.5]])
    bad = torch.tensor([[0.5, -0.5, 0.2, -1.0]])
    s_good = model.score(good)
    s_bad = model.score(bad)
    manual = -torch.log(torch.sigmoid(s_good - s_bad)).mean()
    batch = {"good": good, "bad": bad}

    assert torch.isclose(model.dpo_loss(s_good, s_bad), manual, atol=1e-6)
    assert torch.isclose(model.step(batch), manual, atol=1e-6)


def test_return_window_extractor():
    extractor = ReturnWindowExtractor(lookback=5)
    prices = np.array([100, 101, 99, 102, 103, 105], dtype=np.float32)
    feat = extractor(prices, idx=5)
    # Should have length lookback-1 = 4
    assert len(feat) == 4
    # Validate first element (pad zeros)
    assert math.isclose(float(feat[-1]), math.log(105 / 103), rel_tol=1e-6)


def test_dataset_pair_shapes(tmp_path):
    # Create tiny synthetic data
    csv = tmp_path / "ABC.csv"
    prices = np.arange(10, dtype=np.float32) + 100
    import pandas as pd

    pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=10), "Close": prices}).to_csv(
        csv, index=False
    )
    # Parquet with one pair (idx_good=8, idx_bad=2)
    pq = tmp_path / "pairs.parquet"
    df_pairs = pd.DataFrame({"symbol": ["ABC"], "idx_good": [8], "idx_bad": [2]})
    df_pairs.to_parquet(pq, index=False)

    ds = PreferencePairDataset(pairs_file=pq, prices_dir=tmp_path, lookback=3)
    sample = ds[0]
    assert sample["good"].shape == sample["bad"].shape == torch.Size([2])
