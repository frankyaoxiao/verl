FROM e2bdev/code-interpreter:latest

# Ensure noninteractive apt operations
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        pkg-config \
        curl \
        ca-certificates \
        bzip2 \
        wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget -O /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -b -p /opt/miniconda3 && \
    rm /tmp/miniconda.sh

# Make conda available in all shells (support both /opt/conda and /opt/miniconda3 layouts)
RUN /bin/bash -lc " \
    if [ -f /opt/conda/etc/profile.d/conda.sh ]; then \
        CONDA_PREFIX=/opt/conda; \
    elif [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then \
        CONDA_PREFIX=/opt/miniconda3; \
        ln -s /opt/miniconda3 /opt/conda || true; \
    else \
        echo 'Could not find conda installation' >&2; \
        exit 1; \
    fi; \
    echo \". ${CONDA_PREFIX}/etc/profile.d/conda.sh\" >> /etc/profile.d/conda.sh; \
    echo 'conda activate base' >> /etc/profile.d/conda.sh \
"

ENV PATH=/opt/conda/bin:/opt/miniconda3/bin:$PATH \
    CONDA_DEFAULT_ENV=base

# Pre-create a default working directory matching SWEbench expectations
RUN mkdir -p /workspace && chmod -R 777 /workspace
WORKDIR /workspace
