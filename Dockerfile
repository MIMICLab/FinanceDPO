# ---------------------------------------------------------------
# Dockerfile – Reproducible GPU/CPU environment for FinanceDPO
# ---------------------------------------------------------------
# Specify a different BASE_IMAGE at build time to switch between GPU / CPU.
#   docker build --build-arg BASE_IMAGE=ubuntu:22.04 -t dpo-finance:cpu .
# Defaults to CUDA 12.4.1 for GPU training.
ARG BASE_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

# 1. System dependencies
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 2. Conda installation (Miniconda‑lite)
ENV CONDA_DIR=/opt/conda
RUN curl -Lq https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && rm miniconda.sh && \
    $CONDA_DIR/bin/conda clean -afy
ENV PATH=$CONDA_DIR/bin:$PATH

# 3. Copy repo
WORKDIR /workspace/dpo_financial_repo
COPY . .

# 4. Create env from environment.yml
RUN conda env create -f environment.yml && \
    echo "source activate dpo-finance" > ~/.bashrc
ENV PATH=$CONDA_DIR/envs/dpo-finance/bin:$PATH

# 5. Install project (editable)
RUN pip install -e .

# ensure project is importable without installing again in dev shells
ENV PYTHONPATH=/workspace/dpo_financial_repo:${PYTHONPATH} \
    PYTHONUNBUFFERED=1

# 6. Default command – interactive shell
CMD ["/bin/bash"]