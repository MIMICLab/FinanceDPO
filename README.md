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
python src/dpo_forecasting/data/download.py \
    --symbols-file data/sp500.txt \
    --start 2000-01-01
```
> `data/sp500.txt` is a plain-text list of ticker symbols, one per line (e.g. `AAPL`, `MSFT`, …).

### 3. Generate preference pairs
```bash
python src/dpo_forecasting/data/make_pairs.py \
    --prices-dir data/raw \
    --lookahead 7  --lookback 365 \
    --good-quantile 0.8 --bad-quantile 0.2 \
    --cache-file data/pairs_cache.pt \
    --skip-parquet             # (선택) Parquet 건너뛰기
```
> If you supply `--skip-parquet`, only the Torch cache (`pairs_cache.pt`) is written.  
> Keep the `--skip-parquet` flag **off** if you still want a `.parquet` file for evaluation utilities that expect it.

### 4. Train a DPO model
```bash
python src/train.py \
   dataset.cache_file=data/pairs_cache.pt \
   trainer.max_epochs=30
```

### 5. Evaluate & back-test
```bash
# 5‑a. Pairwise accuracy / ROC
python src/eval.py --checkpoint latest.ckpt \
                   --pairs-file data/pairs.parquet     # or --cache-file

# 5‑b. Quick daily long/short strategy
python src/backtest.py --checkpoint latest.ckpt \
                       --prices-dir data/raw

# 5‑c. Advanced back‑test (position sizing, stop‑loss, leverage, cost)
python src/advanced_backtest.py --checkpoint latest.ckpt \
                                --prices-dir data/raw \
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
python src/train.py \
    dataset.cache_file=data/sp500_pairs_cache.pt \
    trainer.max_epochs=20 \
    +train.kl_coeff=0           # KL off

# 7‑b. Fine‑tune on AAPL only, pulling against the frozen base net
python src/train.py \
    dataset.cache_file=data/aapl_pairs_cache.pt \
    +model.init_from=runs/sp500_base.ckpt \
    +train.kl_coeff=0.05        # small KL to retain global knowledge
```

The first command produces `runs/sp500_base.ckpt`.  
The second command loads that checkpoint as `reference_net`, freezes it
internally, and trains a new policy that specializes on AAPL while being softly
regularized toward the base model.

### 7. Hyperparameter optimization — end‑to‑end (pairs + training)

The file `configs/hpo.yaml` defines a search space for **lookback / lookahead**,
model size, learning rate, etc.  
A tiny pipeline script regenerates preference pairs **and** trains the model for
each trial, so you can safely vary `dataset.lookback` and `lookahead`:

```bash
python -m src.hpo_pipeline -m +hpo=default
```

* `-m` asks **Hydra** to launch a **multirun** (one process per Optuna trial).  
* The pipeline calls  
  1) `data.make_pairs` → creates a unique cache file  
  2) `train.py` → trains on that cache and prints `val_sharpe`  
* Optuna maximizes the first metric logged with `prog_bar=True`; we recommend
  `val_sharpe`.

Results are stored under `multirun/YYYY-MM-DD/HH-MM-SS/{0,1,2,…}/`.  
Open `optuna.log` there to see the best hyper‑parameters.

> **Tip :** edit `configs/hpo.yaml` to add or widen ranges, e.g.  
> `dataset.lookback: "choice(21, 31, 61, 91)"`.
  


---

## 📂 Project layout
```
FinanceDPO/
├── data/                  # raw & processed data (git-ignored)
├── docs/                  # extended docs / diagrams
├── notebooks/             # exploratory notebooks
├── src/
│   └── dpo_forecasting/
│       ├── data/          # data loaders & pair builders
│       ├── models/        # model definitions & loss functions
│       └── configs/       # Hydra configs (dpo.yaml, hpo.yaml …)
├── tests/                 # unit tests
├── environment.yml        # conda spec
├── pyproject.toml         # install package via `pip install -e .`
├── .github/               # CI workflows
└── README.md
```

---

## 🧑‍💻 Contributing

We welcome pull requests! See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the workflow.

---

## 📜 License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.
