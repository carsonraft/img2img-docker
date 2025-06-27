ARG BASE_IMAGE=nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
FROM ${BASE_IMAGE} AS dev-base

ARG MODEL_URL
ENV MODEL_URL=${MODEL_URL}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    SHELL=/bin/bash

# Install just Python first - minimal setup
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends python3.10 python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install minimal packages needed for model download
RUN pip3 install diffusers transformers accelerate safetensors

WORKDIR /opt/ckpt

# Copy and run model fetcher FIRST - absolute fail fast
COPY model_fetcher.py /opt/ckpt/model_fetcher.py
RUN python3 model_fetcher.py --model_url=${MODEL_URL}
RUN echo "Model download completed successfully!"

# Now do the rest of the system setup
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt-get install --yes --no-install-recommends \
    wget \
    bash \
    openssh-server \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

# Install remaining Python dependencies
RUN pip3 install torch==2.0.0 torchvision
COPY requirements.txt /opt/ckpt/requirements.txt
RUN pip3 install -r /opt/ckpt/requirements.txt

# Copy remaining files
COPY . /opt/ckpt

CMD python3 -u /opt/ckpt/runpod_infer.py --model_url="$MODEL_URL"
