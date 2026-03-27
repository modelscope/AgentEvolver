#!/bin/bash

set -e
set -o pipefail

# Path detection
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
BEYONDAGENT_DIR="$(dirname "$(dirname "$ENV_SERVICE_DIR")")"
BFCL_ROOT="$SCRIPT_DIR"
WORKSPACE_DIR="$BEYONDAGENT_DIR"


# 2. 环境变量配置
echo "📁 设置环境变量..."
export NODE_ENV=production
export WORKSPACE_DIR="$WORKSPACE_DIR"
export BFCL_ROOT="$BFCL_ROOT"
export PYTHONPATH="$BEYONDAGENT_DIR:$PYTHONPATH"


# 3. Conda 环境创建
if ! conda info --envs | grep -w "bfcl" &>/dev/null; then
    echo "🐍 创建 Conda 环境 bfcl（Python 3.11.13）..."
    conda create -n bfcl python=3.11.13 -y
else
    echo "⚠️ Conda 环境 bfcl 已存在，请删除或修改。（本次已跳过创建）。"
fi

# 4. 安装依赖
if [ -d "$SCRIPT_DIR/gorilla" ]; then
    echo "🔄 更新 gorilla 仓库..."
    cd "$SCRIPT_DIR/gorilla"
    git pull
else
    echo "📦 克隆 gorilla 仓库..."
    git clone https://github.com/ShishirPatil/gorilla.git
fi

echo "📋 安装 Python 依赖..."

conda run -n bfcl pip install -e "$SCRIPT_DIR/gorilla/berkeley-function-call-leaderboard/."
conda run -n bfcl pip install -r "$SCRIPT_DIR/requirements.txt"

# 5. 准备数据
echo "📁 准备 BFCL 数据..."
cp -r "$SCRIPT_DIR/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data" "$SCRIPT_DIR/bfcl_eval"

cd "$SCRIPT_DIR/"
echo "当前工作目录: $(pwd)"
echo "脚本目录: $SCRIPT_DIR"

python "$SCRIPT_DIR/bfcl_dataprocess.py"

# 6. 设置环境变量
echo "🌎 设置环境变量..."
export ENV_PATH="$SCRIPT_DIR"
export BFCL_DATA_PATH="$ENV_PATH/bfcl_data/multi_turn_base_processed.jsonl"
export BFCL_SPLID_ID_PATH="$ENV_PATH/bfcl_data/multi_turn_base_split_ids.json"
export BFCL_ANSWER_PATH="$ENV_PATH/bfcl_eval/possible_answer"
export OPENAI_API_KEY="$OPENAI_API_KEY"



echo "✅ 设置完成！"

echo ""


echo "如果需要更换数据集，可以使用以下命令："
echo ""
echo "可选的数据集有:"
echo "- all"
echo "- all_scoring"
echo "- multi_turn"
echo "- single_turn"
echo "- live"
echo "- non_live"
echo "- non_python"
echo "- python"
echo "- multi_turn_base"
echo ""
echo "示例命令:"
echo "export DATASET_NAME=multi_turn_base"
echo "export BFCL_DATA_PATH=\"$ENV_PATH/bfcl_data/\${DATASET_NAME}_processed.jsonl\""
echo "export BFCL_SPLID_ID_PATH=\"$ENV_PATH/bfcl_data/\${DATASET_NAME}_split_ids.json\""
echo ""
echo "将 \$DATASET_NAME 替换为您想要使用的数据集名称即可。"

echo "👉 启动方法："
echo "----------------------------------------"
echo "source \$(conda info --base)/etc/profile.d/conda.sh"
echo "conda activate bfcl"
echo "cd $BEYONDAGENT_DIR/env_service/launch_script"
echo "bash bfcl.sh"
echo "----------------------------------------"

exec bash
