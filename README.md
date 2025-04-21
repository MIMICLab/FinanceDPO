# DPO‑Finance

**Direct Preference Optimization for Decision‑Driven Financial Time‑Series Forecasting**

---

## ✨ What’s inside?

* **Opinionated Python project layout** – `src/`, `notebooks/`, `tests/`, `docs/`
* **Ready‑to‑run training script** – [`src/train.py`](src/train.py) wired for **Hydra** + **PyTorch Lightning**
* **Conda environment spec** – [`environment.yml`](environment.yml) for reproducible installs
* **CI & style tooling** – `.pre‑commit‑config.yaml`, `.gitignore`, GitHub Actions, `pytest` placeholder
* **Developer docs** – this `README.md`, [`CONTRIBUTING.md`](CONTRIBUTING.md), MIT [`LICENSE`](LICENSE)

---

## 🔧 Quick start

### 1. Clone & set up
```bash
# Clone
git clone https://github.com/yourname/dpo_financial_repo.git
cd dpo_financial_repo

# Create environment
conda env create -f environment.yml
conda activate dpo-finance
# (Optional) Install pre‑commit hooks
pre-commit install
```

### 2. Download raw price data (example: S&P 500)
```bash
python src/dpo_forecasting/data/download.py \
    --symbols-file data/sp500.txt \
    --start 2000-01-01
```

### 3. Generate preference pairs
```bash
python src/dpo_forecasting/data/make_pairs.py \
   --prices-dir data/raw \
   --lookahead 20 \
   --out-file data/pairs.parquet
```

### 4. Train a DPO model
```bash
python src/train.py \
   dataset.pairs_file=data/pairs.parquet \
   dataset.prices_dir=data/raw \
   trainer.max_epochs=30
```

### 5. Evaluate & back‑test
```bash
# Pairwise metrics
python src/eval.py --checkpoint runs/latest.ckpt \
                   --pairs-file data/pairs.parquet --prices-dir data/raw
# Simple daily long/short strategy
python src/backtest.py --checkpoint runs/latest.ckpt --prices-dir data/raw
```

---

## 📂 Project layout
```
dpo_financial_repo/
├── data/                  # raw & processed data (git‑ignored)
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

## 🧑‍💻 Contributing

We welcome pull requests! See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the workflow.

---

## 📜 License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.
