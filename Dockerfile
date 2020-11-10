FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   wget \
                   ca-certificates \
                   python3 \
                   python3-pip && \
    rm -rf /var/lib/apt/lists

# RUN python3 -m pip install --no-cache-dir --upgrade pip && \
#     python3 -m pip install --no-cache-dir \
#     mkl \
#     torch

# RUN git clone https://github.com/NVIDIA/apex
# RUN cd apex && \
#     python3 setup.py install && \
#     pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# INSTALLATION OF CONDA
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

COPY . /screenwriter

RUN conda env create -f /screenwriter/environment.yml
RUN echo "source activate script" > ~/.bashrc
ENV PATH /opt/conda/envs/script/bin:$PATH

CMD tail -f /dev/null
