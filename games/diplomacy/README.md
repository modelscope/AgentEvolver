# Diplomacy Web Game

本项目为《Diplomacy》博弈游戏的多智能体仿真与 Web 交互平台，支持 AI 代理与人类玩家参与，具备灵活的配置与可视化界面。

## 目录结构

- `engine.py` / `game.py` / `utils.py`：核心游戏逻辑与工具
- `agents/`：各类智能体实现（AI、终端用户、回显等）
- `prompt/`：多语言提示词（英文/中文），支持自定义与扩展
- `web/`：Web 服务端与前端，包括：
  - `server.py`：FastAPI 后端，API接口
  - `run_web_game.py`：Web 启动入口
  - `game_state_manager.py`：游戏状态管理
  - `static/`：前端静态资源（HTML/CSS/JS）

## 快速启动

1. 安装依赖

```bash
pip install agentscope
pip install diplomacy
```

2. 启动 Web 服务

```bash
python games/diplomacy/web/run_web_game.py
```

3. 访问界面

浏览器打开：

```
http://localhost:8000
```

## 主要功能

- 多智能体博弈仿真，支持 AI/人类混合参与
- 支持多种模型与角色配置
- 可视化地图与交互式日志
- 多语言提示词（英文/中文）
- 灵活参数配置（回合数、谈判轮次、地图等）

## 配置说明

- 启动前可在前端配置：模式、角色、模型、最大回合、谈判轮次、语言等
- 支持 observe（旁观）与 participate（参与）两种模式

## 贡献与开发

欢迎提交 issue 或 pull request 进行功能扩展与优化。

## 许可证

本项目遵循 Apache License 2.0。
