# ------------------------------------------------------------
# Makefile — Common developer commands for DPO‑Finance project
# ------------------------------------------------------------

# Variables (override with make VAR=value)
ENV ?= dpo-finance
PYTHON = conda run -n $(ENV) python

# Helper directories / files
PAIRS ?= data/pairs.parquet
PRICES ?= data/raw
CKPT ?= runs/latest.ckpt

.PHONY: help env download pairs train eval backtest test lint format docker

help:
	@echo "Available targets:"
	@echo "  env       – create/activate Conda env ($(ENV))"
	@echo "  download  – download price data (requires SYMBOLS=txt,start,end)"
	@echo "  pairs     – generate preference pairs (requires PRICES, LOOKAHEAD)"
	@echo "  train     – train model (results in runs/)"
	@echo "  eval      – pairwise accuracy on $(PAIRS)"
	@echo "  backtest  – run daily long/short back‑test"
	@echo "  test      – pytest unit tests"
	@echo "  lint      – flake8 lint"
	@echo "  format    – black auto‑format"
	@echo "  docker    – build Docker image (tag: dpo-finance)"

# ------------------------------------------------------------

env:
	conda env create -f environment.yml || true
	@echo "Activate with: conda activate $(ENV)"

download:
	$(PYTHON) src/dpo_forecasting/data/download.py \
		--symbols-file $(SYMBOLS) --start $(START) --end $(END) --out-dir $(PRICES)

pairs:
	$(PYTHON) src/dpo_forecasting/data/make_pairs.py \
		--prices-dir $(PRICES) --lookahead $(LOOKAHEAD) --out-file $(PAIRS)

train:
	$(PYTHON) src/train.py dataset.pairs_file=$(PAIRS) dataset.prices_dir=$(PRICES)

eval:
	$(PYTHON) src/eval.py --checkpoint $(CKPT) --pairs-file $(PAIRS) --prices-dir $(PRICES)

backtest:
	$(PYTHON) src/backtest.py --checkpoint $(CKPT) --prices-dir $(PRICES)

test:
	pytest -q

lint:
	flake8 src tests

format:
	black src tests

docker:
	docker build -t dpo-finance .
