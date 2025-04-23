# FinanceDPO

**Direct Preference Optimization for Decision-Driven Financial Time-Series Forecasting**

---

## ✨ What’s inside?

* **Opinionated Python project layout** – `src/`, `notebooks/`, `tests/`, `docs/`
* **Ready-to-run training script** – [`src/train.py`](src/train.py) wired for **Hydra** + **PyTorch Lightning**
* **Conda environment spec** – [`environment.yml`](environment.yml) for reproducible installs
* **CI & style tooling** – `.pre-commit-config.yaml`, `.gitignore`, GitHub Actions, `pytest` placeholder
* **Developer docs** – this `README.md`, [`CONTRIBUTING.md`](CONTRIBUTING.md), MIT [`LICENSE`](LICENSE)

---

## 🔧 Quick start

### 1. Clone & set up
```bash
# Clone
git clone https://github.com/MIMICLab/FinanceDPO.git
cd FinanceDPO

# Create environment
conda env create -f environment.yml
conda activate dpo-finance

> **Python version**: `environment.yml` pins `python=3.11`.  
> If your base conda uses a different version, run  
> `conda env create -f environment.yml -n dpo-finance`.
```

### 2. Download raw price data (example: S&P 500)
```bash
python src/dpo_forecasting/preprocessing/download.py \
    --symbols-file data/sp500.txt \
    --start 2000-01-01
```
> `data/sp500.txt` is a plain-text list of ticker symbols, one per line (e.g. `AAPL`, `MSFT`, …).

### 3. Generate preference pairs
```bash
python src/dpo_forecasting/preprocessing/make_pairs.py \
    --config src/dpo_forecasting/configs/findpo_sp500_base.yaml # or other configs
```

### 4. Train a DPO model
```bash
python src/train.py \
    --config src/dpo_forecasting/configs/findpo_sp500_base.yaml # or other configs
```

### 5. Evaluate & back-test
```bash
# 5‑a. Pairwise accuracy / ROC
python src/eval.py --checkpoint latest.ckpt \
                   --pairs-file data/pairs.parquet     # or --cache-file

# 5‑b. Quick daily long/short strategy
python src/backtest.py --checkpoint latest.ckpt \
                       --prices_dir data/raw

# 5‑c. Advanced back‑test (position sizing, stop‑loss, leverage, cost)
python src/advanced_backtest.py --checkpoint latest.ckpt \
                                --prices_dir data/raw \
                                --lookback 31 --hold 5 \
                                --score-thresh 0.5 \
                                --max-long 0.3 --max-short 0.1 \
                                --cost-bps 5
```
`advanced_backtest.py` applies position‑scaled weights, transaction costs,
and optional equity stops, giving a closer approximation to live
portfolio behaviour.  Adjust `--hold`, `--max-long`, `--cost-bps` and other
flags to reflect your trading assumptions.

### 6. Fine‑tune a single symbol with a frozen reference net

You can first train a **broad, market‑wide model** (e.g. on all S&P 500 pairs),
then fine‑tune it for a specific symbol while keeping the original behaviour
as a reference via a small KL penalty:

```bash
# 7‑a. Train a base model on the full S&P 500 cache
python src/train.py --config src/configs/findpo_sp500_base.yaml

# After saving latest ckpt stored in tb_logs/version_x/checkpoints
cp tb_logs/dpo/version_x/checkpoints/xxx.ckpt findpo_sp500_base.ckpt

# 7‑b. Fine‑tune on MAG7, pulling against the frozen base net
python src/train.py --config src/configs/findpo_mag7_base.yaml

```

The first command produces `runs/sp500_base.ckpt`.  
The second command loads that checkpoint as `reference_net`, freezes it
internally, and trains a new policy that specializes on AAPL while being softly
regularized toward the base model.

---

## 📂 Project layout
```
FinanceDPO/
├── data/                  # raw & processed data (git-ignored)
├── src/
│   └── dpo_forecasting/
│       ├── data/          # data loaders & pair builders
│       ├── models/        # model definitions & loss functions
│       └── configs/       # Hydra configs (dpo.yaml, hpo.yaml …)
├── tests/                 # unit tests
├── environment.yml        # conda spec
├── pyproject.toml         # install package via `pip install -e .`
└── README.md
```

---

## 🧑‍💻 Contributing

We welcome pull requests! See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the workflow.

---

## 📜 License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.
