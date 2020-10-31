FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   ca-certificates \
                   python3 \
                   python3-pip && \
    rm -rf /var/lib/apt/lists

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    mkl \
    torch

RUN git clone https://github.com/NVIDIA/apex
RUN cd apex && \
    python3 setup.py install && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# INSTALLATION OF CONDA
ARG PYTHON_VERSION=3.7
ARG CONDA_VERSION=3
ARG CONDA_PY_VERSION=4.5.11

ENV PATH /opt/conda/bin:$PATH
RUN wget — quiet https://repo.anaconda.com/miniconda/Miniconda$ CONDA_VERSION-$ CONDA_PY_VERSION-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo “. /opt/conda/etc/profile.d/conda.sh” >> ~/.bashrc && \
    echo “conda activate base” >> ~/.bashrc

ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

# RUN conda update -n base -c defaults conda && \
#     conda env create -f /screenwriter/environment.yml
RUN conda env create -f /screenwriter/environment.yml
ENV PATH /opt/conda/envs/script/bin:$PATH

WORKDIR /screenwriter
