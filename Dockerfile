# ---------------------------------------------------------------
# Dockerfile – Reproducible GPU/CPU environment for DPO‑Finance
# ---------------------------------------------------------------
    FROM nvidia/cuda:12.4.1-devel-ubuntu22.04  # change to non‑CUDA Ubuntu if CPU‑only

    # 1. System dependencies
    RUN apt-get update && apt-get install -y --no-install-recommends \
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
    
    # 6. Default command – interactive shell
    CMD ["/bin/bash"]
    