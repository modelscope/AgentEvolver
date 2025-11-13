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

RUN conda create -n bfcl python=3.11.13 -y

COPY env_service/ ./env_service/
RUN apt-get install gcc python3-dev -y

# 克隆 gorilla 仓库
RUN git clone https://github.com/ShishirPatil/gorilla.git

# 安装依赖
RUN conda run -n bfcl pip install -e ./gorilla/berkeley-function-call-leaderboard/. && \
    conda run -n bfcl pip install -r ./env_service/environments/bfcl/requirements.txt

# 准备数据
RUN cp -r ./gorilla/berkeley-function-call-leaderboard/bfcl_eval/data ./env_service/environments/bfcl/bfcl_eval

# 切换到脚本目录并处理数据
WORKDIR /BeyondAgent/env_service/environments/bfcl
RUN python bfcl_dataprocess.py

# 设置环境变量
ENV PYTHONPATH=/BeyondAgent:$PYTHONPATH
ENV ENV_PATH=/BeyondAgent/env_service/environments/bfcl
ENV BFCL_DATA_PATH=$ENV_PATH/bfcl_data/multi_turn_base_processed.jsonl
ENV BFCL_SPLID_ID_PATH=$ENV_PATH/bfcl_data/multi_turn_base_split_ids.json
ENV BFCL_ANSWER_PATH=$ENV_PATH/bfcl_eval/possible_answer

ENV OPENAI_API_KEY=$OPENAI_API_KEY

WORKDIR /BeyondAgent

CMD . /opt/conda/etc/profile.d/conda.sh && \
    conda activate bfcl && \
    cd /BeyondAgent/env_service/launch_script && \
    exec bash bfcl.sh
