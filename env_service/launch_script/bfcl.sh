#!/bin/bash
set -eu

# 注意：
# 请先执行文件：EnvService/env_sandbox/environments/bfcl/bfcl_dataprocess.py
# 获取BFCL_DATA_PATH与BFCL_SPLID_ID_PATH，并对应设置以上两个变量

# 环境变量（请改成实际路径）

#
# 获取 launch_script 的目录
LAUNCH_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 获取 env_service 的目录
ENV_SERVICE_DIR="$(dirname "$LAUNCH_SCRIPT_DIR")"

# 获取 bfcl 环境的目录
DEFAULT_BFCL_ENV_DIR="$ENV_SERVICE_DIR/environments/bfcl"
BFCL_ENV_DIR=${BFCL_ENV_DIR:-$DEFAULT_BFCL_ENV_DIR}

export ENV_PATH="$BFCL_ENV_DIR"
export BFCL_DATA_PATH="$BFCL_ENV_DIR/bfcl_data/multi_turn_base_processed.jsonl"
export BFCL_SPLID_ID_PATH="$BFCL_ENV_DIR/bfcl_data/multi_turn_base_split_ids.json"
export BFCL_ANSWER_PATH="$BFCL_ENV_DIR/bfcl_eval/possible_answer"

echo "🌍 环境变量设置完成:"
echo "ENV_PATH: $ENV_PATH"
echo "BFCL_DATA_PATH: $BFCL_DATA_PATH"
echo "BFCL_SPLID_ID_PATH: $BFCL_SPLID_ID_PATH"
echo "BFCL_ANSWER_PATH: $BFCL_ANSWER_PATH"

# 检查文件是否存在
if [ -f "$BFCL_DATA_PATH" ]; then
    echo "✅ 数据文件存在: $BFCL_DATA_PATH"
else
    echo "❌ 数据文件不存在: $BFCL_DATA_PATH"
fi

if [ -f "$BFCL_SPLID_ID_PATH" ]; then
    echo "✅ 分割ID文件存在: $BFCL_SPLID_ID_PATH"
else
    echo "❌ 分割ID文件不存在: $BFCL_SPLID_ID_PATH"
fi

if [ -d "$BFCL_ANSWER_PATH" ]; then
    echo "✅ 答案文件夹存在: $BFCL_ANSWER_PATH"
else
    echo "❌ 答案文件夹不存在: $BFCL_ANSWER_PATH"
fi

export OPENAI_API_KEY=xx

# only for multinode running
export RAY_ENV_NAME=bfcl 

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 导航到项目根目录 (envservice)
PROJECT_ROOT="$SCRIPT_DIR/../../"
cd "$PROJECT_ROOT"

# 设置 PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# 打印当前工作目录和 PYTHONPATH 以进行调试
echo "Current working directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"

# 运行 Python 命令
exec python -m env_service.env_service --env bfcl --portal 127.0.0.1 --port 8080