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
python -m dpo_forecasting.data.make_pairs \
    --prices-dir data/raw \
    --lookahead 7  --lookback 31 \
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
# (Uses Hydra: default config is configs/dpo.yaml; override with +key=value)
```

### 5. Evaluate & back-test
```bash
# Pairwise metrics
python src/eval.py --checkpoint latest.ckpt \
                   --pairs-file data/pairs.parquet 
# Simple daily long/short strategy
python src/backtest.py --checkpoint latest.ckpt --prices-dir data/raw
```
> Tip: save checkpoints with a timestamp to avoid accidental overwrites.

If you trained from the cache only (skipped the Parquet), pass `--cache-file` instead of `--pairs-file` to `eval.py`.

### 6. Hyperparameter optimization (Optuna + Hydra)

A ready‑made config file `configs/hpo.yaml` lets you launch a multirun sweep with Optuna:

```bash
python src/train.py -m +hpo
```

* `-m` tells **Hydra** to execute a **multirun**, spawning one process per trial.  
* `+hpo` overrides the default settings with `configs/hpo.yaml`, which  
  – switches the Hydra **sweeper** to Optuna,  
  – sets `n_trials`, search spaces, and the target metric.

Trials are saved under `multirun/DATE_TIME/{0,1,2,…}/`.  
Open `optuna.log` in that folder to see the best parameters and their scores.

To customise the search space or the number of trials, edit `configs/hpo.yaml`:

```yaml
hydra:
  sweeper:
    n_trials: 50        # increase for a wider search
params:
  train.lr:          "loguniform(1e-5,1e-3)"
  model.hidden_dim:  "choice(128,256,512)"
```

The sweeper maximizes the **first** metric that you log with `prog_bar=True`; we recommend logging a risk‑adjusted metric such as `val_sharpe`.

  

### 7. Fine‑tune a single symbol with a frozen reference net

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
