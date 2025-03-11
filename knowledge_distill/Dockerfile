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
RUN git clone --depth=1 https://github.com/DarthIV02/OFAKD.git /home/OFAKD
RUN git clone --depth=1 https://github.com/DarthIV02/HyperLiDAR.git /home/HyperLiDAR
RUN conda create -n env python=3.8 -y && \
    conda init bash

WORKDIR /home/HyperLiDAR

# make RUN commands use the new environment
SHELL ["conda", "run", "--no-capture-output", "-n", "env", "/bin/bash", "-c"]

RUN echo 'conda activate env' >> /root/.bashrc

RUN echo Y | pip install --force-reinstall torch==1.12.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

RUN bash nautilus/3dlabelprop_setup.sh

RUN git clone https://github.com/valeoai/WaffleIron

RUN echo Y | pip install -e WaffleIron/

RUN wget https://github.com/valeoai/ScaLR/releases/download/v0.1.0/info_datasets.tar.gz

RUN tar -xvzf info_datasets.tar.gz

RUN rm info_datasets.tar.gz

RUN echo Y | pip install -r requirements.txt

WORKDIR /home/OFAKD

RUN echo Y | pip install pyyaml

RUN echo Y | pip install timm==0.6.5 && \
    echo Y | pip install -r requirements.txt

# echo Y | pip install --force-reinstall torch==0.11.0 torchvision==0.11.3 torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 && \

RUN apt install software-properties-common

RUN echo Y | add-apt-repository ppa:ubuntu-toolchain-r/test

RUN echo Y | apt-get update

RUN echo Y | apt-get upgrade libstdc++6

RUN echo Y | apt-get dist-upgrade

WORKDIR /home/HyperLiDAR

RUN RUN echo Y | pip install --force-reinstall torch==1.12.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113