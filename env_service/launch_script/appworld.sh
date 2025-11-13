#!/bin/bash


# 可以修改为自己的appworld数据路径
# 使用安装的默认Path detection
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
BEYONDAGENT_DIR="$(dirname "$(dirname "$ENV_SERVICE_DIR")")"
APPWORLD_ROOT="${APPWORLD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../environments/appworld" && pwd)}"
export APPWORLD_ROOT
echo "APPWORLD_ROOT: $APPWORLD_ROOT"

#
export RAY_ENV_NAME=appworld


# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 导航到项目根目录 (env_service)
PROJECT_ROOT="$SCRIPT_DIR/../../"
cd "$PROJECT_ROOT"

# 设置 PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 打印当前工作目录和 PYTHONPATH 以进行调试
echo "Current working directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"

# 运行 Python 命令
exec python -m env_service.env_service --env appworld --portal 127.0.0.1 --port 8080