FROM node:22-slim

# Set ENV variables
ENV NODE_ENV=production
ENV WORKSPACE_DIR=/workspace

ARG DEBIAN_FRONTEND=noninteractive
RUN mkdir -p /etc/apt && touch /etc/apt/sources.list
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    git \
    gcc \
    python3-dev \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O /tmp/miniconda.sh && \
    chmod +x /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Optionally, add conda to PATH
ENV PATH="/opt/conda/bin:${PATH}"

# Initialize conda for shell interaction (optional)
RUN /opt/conda/bin/conda init bash

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

WORKDIR /BeyondAgent

RUN conda create -n appworld python=3.11.0 -y && \
    conda install -n appworld -y libcst

COPY env_service/ ./env_service/

RUN ls -l ./env_service/environments/appworld
RUN apt-get install gcc python3-dev -y

RUN conda run -n appworld pip install -r ./env_service/environments/appworld/requirements.txt

RUN mkdir appworld && cd appworld

RUN /opt/conda/envs/appworld/bin/appworld install

ENV APPWORLD_ROOT=/BeyondAgent/appworld/

RUN /opt/conda/envs/appworld/bin/appworld download data

RUN mv /BeyondAgent/data /BeyondAgent/appworld/
# If download fails, you may get the data from this source and move it locally by commenting two lines above and uncommenting the following lines

# RUN wget -O appworld_data.zip "https://dail-wlcb.oss-accelerate.aliyuncs.com/eric.czq/appworld_data.zip"

# RUN mkdir -p /BeyondAgent/appworld && \
#     mkdir /tmp/unziptemp && \
#     unzip appworld_data.zip -d /tmp/unziptemp && \
#     mv /tmp/unziptemp/*/* /BeyondAgent/appworld && \
#     rm -rf /tmp/unziptemp


ENV PYTHONPATH=/BeyondAgent:$PYTHONPATH

WORKDIR /BeyondAgent

CMD . /opt/conda/etc/profile.d/conda.sh && \
    conda activate appworld && \
    cd /BeyondAgent/env_service/ && \
    exec bash appworld.sh
