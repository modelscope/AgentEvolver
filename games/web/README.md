# Web 游戏界面

统一的 Web 界面，支持 Avalon 和 Diplomacy 两个游戏的观战和参与模式。

## 项目结构

```
games/web/
├── server.py                 # FastAPI 服务器：HTTP/WebSocket 端点
├── game_state_manager.py     # 游戏状态管理器：管理状态、消息队列、WebSocket 连接
├── web_agent.py              # Web 代理：WebUserAgent（参与模式）、ObserveAgent（观战模式）
├── web_user_input.py         # 用户输入处理器：通过 WebSocket 接收前端输入
├── run_web_game.py           # 游戏启动器：在后台线程中运行游戏
├── web_config.yaml           # Web 配置：15 个角色的模型绑定
└── static/                   # 前端静态文件
    ├── index.html            # 主页面：选择角色、游戏、配置
    ├── character_config.html # 角色配置页面：配置每个角色的模型和 API
    ├── main.js               # 主页面逻辑
    ├── avalon/               # Avalon 游戏页面
    └── diplomacy/            # Diplomacy 游戏页面
```

## 后端架构

### 核心组件

#### 1. `server.py` - FastAPI 服务器

**主要端点：**

- `GET /` - 返回 index.html
- `GET /avalon/observe`, `/avalon/participate` - Avalon 游戏页面
- `GET /diplomacy/observe`, `/diplomacy/participate` - Diplomacy 游戏页面
- `GET /api/options?game={game}` - 获取游戏配置选项
  - 无 `game` 参数：返回 `web_config.yaml`（角色名字等）
  - `game=avalon`：返回 Avalon 默认配置
  - `game=diplomacy`：返回 Diplomacy 配置（powers, models 等）
- `POST /api/start-game` - 启动游戏
- `POST /api/stop-game` - 停止游戏
- `GET /api/history` - 获取 Diplomacy 历史记录列表
- `GET /api/history/{index}` - 获取 Diplomacy 历史记录项
- `WebSocket /ws` - WebSocket 连接：实时通信

**调用流程：**

```python
# 启动游戏
POST /api/start-game
  → start_game_thread()  # 在后台线程启动
    → run_avalon() 或 run_diplomacy()  # 运行游戏逻辑
      → 创建 agents（WebUserAgent 或 ThinkingReActAgent）
      → 调用游戏引擎（avalon_game 或 diplomacy_game）
```

#### 2. `game_state_manager.py` - 游戏状态管理器

**核心功能：**

- **状态管理**：统一管理 Avalon 和 Diplomacy 的游戏状态
- **消息队列**：管理用户输入队列和消息广播队列
- **WebSocket 连接管理**：维护所有 WebSocket 连接
- **历史记录**：为 Diplomacy 自动保存历史快照

**关键方法：**

```python
# 用户输入处理
async def put_user_input(agent_id: str, content: str)  # 放入用户输入
async def get_user_input(agent_id: str, timeout: float) -> str  # 获取用户输入

# 消息广播
async def broadcast_message(message: Dict)  # 向所有 WebSocket 连接广播

# 状态更新
def update_game_state(**kwargs)  # 更新游戏状态（自动为 Diplomacy 创建快照）
def save_history_snapshot(kind: str)  # 显式保存历史快照
```

#### 3. `web_agent.py` - Web 代理

**WebUserAgent**（参与模式）：
- 继承自 `UserAgent`
- 使用 `WebUserInput` 处理用户输入
- 在 `observe()` 中广播消息到前端

**ObserveAgent**（观战模式）：
- 继承自 `AgentBase`
- 将所有观察到的消息转发到前端
- 不参与游戏逻辑

#### 4. `web_user_input.py` - 用户输入处理器

**WebUserInput**：
- 实现 `UserInputBase` 接口
- 通过 `GameStateManager` 的队列机制获取用户输入
- 向 WebSocket 发送输入请求，等待前端响应

**调用流程：**

```python
WebUserInput.__call__()
  → broadcast_message(user_input_request)  # 发送请求到前端
  → get_user_input(agent_id)  # 从队列获取用户输入
```

#### 5. `run_web_game.py` - 游戏启动器

**核心函数：**

- `run_avalon()` - 运行 Avalon 游戏
  - 优先使用前端传递的 `agent_configs` 配置
  - 回退到 `web_config.yaml` 中的角色配置
  - 最后使用环境变量作为默认值
  - 根据 `selected_portrait_ids` 创建对应角色的 agent
  - 支持 `preset_roles` 固定角色分配
  
- `run_diplomacy()` - 运行 Diplomacy 游戏
  - 优先使用前端传递的 `agent_configs` 配置
  - 回退到 `config.models` 配置
  - 最后使用环境变量作为默认值
  - 根据 `power_names` 和 `selected_portrait_ids` 创建各 power 的 agent
  - 支持 `power_models` 配置

- `start_game_thread()` - 在后台线程启动游戏
  - 解析前端传入的参数（包括 `agent_configs`）
  - 创建独立的事件循环
  - 在后台线程中运行游戏
  - 支持任务取消（通过 `state_manager._game_task`）

## 前端架构

### 主要页面

#### 1. `index.html` + `main.js` - 主页面

**功能：**
- 显示 15 个角色头像，支持选择/取消
- 选择游戏（Avalon/Diplomacy）和模式（Observer/Participate）
- 配置游戏参数（人数、语言等）
- 预览圆桌布局和角色分配
- 启动游戏

**数据流：**

```javascript
// 初始化
init()
  → loadWebConfig()  // 首次加载：从 /api/options 获取 web_config.yaml
  → renderPortraits()  // 从 localStorage 读取角色配置并渲染

// 选择游戏
setGame(game)
  → fetchDiplomacyOptions()  // 如果是 Diplomacy，从 localStorage 或 API 加载配置

// 启动游戏
buildPayload()  // 构建启动参数
  → POST /api/start-game
  → 跳转到游戏页面
```

**localStorage 存储：**

- `AgentConfigs.v1` - 角色配置（名字、模型、API 等）
- `GameOptions.v1` - 游戏配置（从 `/api/options` 获取）
- `WebConfigLoaded.v1` - 标记是否已加载 web_config.yaml
- `ConfigUpdateTime.v1` - 配置更新时间戳（用于检测更新）

#### 2. `character_config.html` - 角色配置页面

**功能：**
- 配置每个角色的名字、模型、API base、API key
- 保存到 `localStorage` 的 `AgentConfigs.v1`
- 设置 `ConfigUpdateTime.v1` 时间戳，通知其他页面更新

#### 3. 游戏页面（Avalon/Diplomacy）

**Avalon：**
- `observe.html` + `observe.js` - 观战模式
- `participate.html` + `participate.js` - 参与模式

**Diplomacy：**
- `observe.html` + `main.js` - 观战模式（支持历史记录浏览）
- `participate.html` + `main.js` - 参与模式

## 前后端通信

### HTTP API

#### 1. 获取配置选项

```http
GET /api/options?game=diplomacy
```

**响应：**
```json
{
  "powers": ["ENGLAND", "FRANCE", ...],
  "models": ["qwen-plus", "gpt-4o-mini", ...],
  "power_models": {"ENGLAND": "qwen-plus", ...},
  "defaults": {...},
  "default_model": {...}
}
```

#### 2. 启动游戏

```http
POST /api/start-game
Content-Type: application/json

{
  "game": "avalon",
  "mode": "observe",
  "language": "en",
  "num_players": 5,
  "preset_roles": [...],
  "selected_portrait_ids": [1, 2, 3, 4, 5],
  "agent_configs": {
    "1": {"base_model": "gpt-4o-mini", "api_base": "https://api.openai.com/v1", "api_key": "sk-..."},
    "2": {"base_model": "qwen-plus", "api_base": "", "api_key": ""}
  }
}
```

**参数说明：**
- `agent_configs`: 前端传递的角色配置，优先级高于 `web_config.yaml` 和环境变量
  - 键为 `portrait_id`（字符串格式）
  - 值为 `{base_model, api_base, api_key}` 对象

#### 3. 停止游戏

```http
POST /api/stop-game
```

### WebSocket 通信

**连接：**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
```

**消息类型：**

1. **game_state** - 游戏状态更新
```json
{
  "type": "game_state",
  "game": "avalon",
  "status": "running",
  "phase": "mission_voting",
  ...
}
```

2. **message** - 游戏消息（agent 对话）
```json
{
  "type": "message",
  "sender": "Player0",
  "content": "I vote yes",
  "role": "assistant",
  "timestamp": "2024-01-01T00:00:00"
}
```

3. **user_input_request** - 用户输入请求（参与模式）
```json
{
  "type": "user_input_request",
  "agent_id": "player_0_id",
  "prompt": "[Player0] Please provide your input:"
}
```

4. **mode_info** - 模式信息
```json
{
  "type": "mode_info",
  "mode": "participate",
  "user_agent_id": "player_0_id",
  "game": "avalon"
}
```

**前端发送：**

1. **user_input** - 用户输入响应
```json
{
  "type": "user_input",
  "agent_id": "player_0_id",
  "content": "I vote yes"
}
```

## 数据流

### 游戏启动流程

```
前端 (index.html)
  ↓ POST /api/start-game
后端 (server.py)
  ↓ start_game_thread()
后台线程 (run_web_game.py)
  ↓ run_avalon() 或 run_diplomacy()
游戏引擎 (avalon_game / diplomacy_game)
  ↓ 创建 agents
  ↓ 运行游戏逻辑
  ↓ 通过 state_manager.broadcast_message() 发送消息
WebSocket (/ws)
  ↓ 推送到前端
前端游戏页面
  ↓ 显示消息和状态
```

### 用户输入流程（参与模式）

```
游戏引擎
  ↓ WebUserAgent.reply() 需要用户输入
  ↓ WebUserInput.__call__()
  ↓ state_manager.broadcast_message(user_input_request)
WebSocket
  ↓ 推送到前端
前端 (participate.html)
  ↓ 显示输入框
  ↓ 用户输入
  ↓ WebSocket 发送 {type: "user_input", agent_id: "...", content: "..."}
后端 (server.py)
  ↓ _handle_websocket_connection()
  ↓ state_manager.put_user_input()
  ↓ 放入队列
游戏引擎
  ↓ state_manager.get_user_input()
  ↓ 从队列获取
  ↓ 继续游戏逻辑
```

## 配置说明

### web_config.yaml

```yaml
default_model:
  model_name: qwen-plus
  api_base: https://dashscope.aliyuncs.com/compatible-mode/v1
  api_key: ${OPENAI_API_KEY}  # 从环境变量读取

portraits:
  1:
    name: Agent1
    model_name: qwen-max  # 可选：覆盖默认模型
  2:
    name: Agent2
    # 使用 default_model
```

### 环境变量

- `API_KEY` - API 密钥（用于 `${API_KEY}` 占位符）
- `MODEL_NAME` - 默认模型名称
- `AVALON_CONFIG_YAML` - Avalon 配置文件路径
- `DIPLOMACY_CONFIG_YAML` - Diplomacy 配置文件路径
- `LOG_DIR` - 日志目录

## 运行

```bash
# 启动服务器
python games/web/server.py

# 或使用 uvicorn
uvicorn games.web.server:app --host 0.0.0.0 --port 8000
```

访问 `http://localhost:8000` 进入主页面。

## 注意事项

1. **localStorage 管理**：
   - 角色配置存储在 `AgentConfigs.v1`（包含 name、base_model、api_base、api_key）
   - 游戏配置缓存到 `GameOptions.v1`，减少 API 调用
   - 使用时间戳机制（`ConfigUpdateTime.v1`）检测配置更新
   - 首次加载时从 `/api/options` 获取 `web_config.yaml` 并合并角色名字

2. **配置优先级**（Agent 模型配置）：
   - 前端传递的 `agent_configs`（最高优先级）
   - `web_config.yaml` 中的角色配置
   - 环境变量（`MODEL_NAME`、`API_KEY` 等）

3. **WebSocket 连接**：
   - 支持多连接（多个标签页/窗口）
   - 游戏停止后，停止广播新消息

4. **游戏状态**：
   - `waiting` - 等待启动
   - `running` - 运行中
   - `finished` - 已完成
   - `stopped` - 已停止
   - `error` - 错误

5. **历史记录**（仅 Diplomacy）：
   - 观战模式自动保存历史快照
   - 可通过 `/api/history` 和 `/api/history/{index}` 访问

6. **线程管理**：
   - 每个游戏在独立的后台线程中运行
   - 每个线程有独立的事件循环
   - 支持通过 `state_manager._game_task` 取消任务

