# Knowledge distillation

FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND='noninteractive'

## apt installs
RUN apt-get update && \
    apt install -y git wget tmux zip git

## install conda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh && \
    bash Anaconda3-2023.03-1-Linux-x86_64.sh -b

ENV PATH=$PATH:/root/anaconda3/bin/

WORKDIR /home

COPY initial_setup.sh .

RUN bash initial_setup.sh

## install repo / conda environment (repo instructions)
RUN git clone --depth=1 https://github.com/Hao840/OFAKD.git /home/OFAKD
RUN conda create -n env python=3.8 -y && \
    conda init bash

WORKDIR /home/OFAKD

# make RUN commands use the new environment
SHELL ["conda", "run", "--no-capture-output", "-n", "cenet", "/bin/bash", "-c"]

RUN echo 'conda activate env' >> /root/.bashrc

RUN echo Y | pip install --force-reinstall torch==1.10.2 torchvision==0.11.3 torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 && \
    echo Y | pip install timm==0.6.5 && \
    echo Y | pip install -r requirements.txt
