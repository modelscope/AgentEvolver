#!/bin/bash

set -e
set -o pipefail

# Path detection
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
BEYONDAGENT_DIR="$(dirname "$(dirname "$ENV_SERVICE_DIR")")"

WORKSPACE_DIR="$BEYONDAGENT_DIR"


# 2. 环境变量配置
echo "📁 设置环境变量..."
export NODE_ENV=production
export WORKSPACE_DIR="$WORKSPACE_DIR"

export PYTHONPATH="$BEYONDAGENT_DIR:$PYTHONPATH"



# 3. Conda 环境创建
if ! conda info --envs | grep -w "openworld" &>/dev/null; then
    echo "🐍 创建 Conda 环境 openworld（Python 3.11）..."
    conda create -n openworld python=3.11.0 -y
else
    echo "⚠️ Conda 环境 openworld 已存在，请删除或修改。（本次已跳过创建）。"
fi
# 4. 安装依赖

echo "📋 安装 Python 依赖..."
conda run -n openworld pip install -r "$SCRIPT_DIR/requirements.txt"

# 5. 初始化 appworld
echo "📁 初始化 openworld..."


echo ""
echo "👉 启动方法："
echo "----------------------------------------"
echo "source \$(conda info --base)/etc/profile.d/conda.sh"
echo "conda activate openworld"
echo "cd $BEYONDAGENT_DIR/env_service/launch_script"
echo "bash openworld.sh"
echo "----------------------------------------"
