# Arena Leaderboard

A multi-game evaluation system with **persistent storage**, **fair model assignment**, and **consistent benchmarking** across repeated self-play games.



## Highlights ✨

-  **Fair Assignment**: weighted random selection balances game counts across models
-  **Persistent Storage**: thread-safe JSON database with incremental writes (resume anytime)
-  **Multi-Game Support**: Avalon + Diplomacy (easy to extend via lazy loading)
-  **Role Statistics**: per-role win rates (game-specific)
-  **API Rate Limiting**: configurable delay between calls to reduce throttling
-  **Performance Tracking**: keeps lightweight rating + win statistics for comparisons



## Quick Start 🚀

### Supported Games 🎮

- **Avalon**: hidden-role social deduction game focused on persuasion, deception, and team coordination.
- **Diplomacy**: multi-power strategy and negotiation game centered on alliances, bargaining, and long-term planning.Run an Evaluation

```bash
# Avalon
python games/evaluation/leaderboard/run_arena.py \
  --game avalon \
  --config games/games/avalon/configs/arena_config.yaml \
  --num-games 200 \
  --max-workers 10

# Diplomacy
python games/evaluation/leaderboard/run_arena.py \
  --game diplomacy \
  --config games/games/diplomacy/configs/arena_config.yaml \
  --num-games 100 \
  --max-workers 10
```



## Continue Runs / Add Models 🔁➕

The system automatically loads existing leaderboard data and continues updating it.

### 1) Add models in the config

```yaml
arena:
  models:
    - qwen-plus
    - qwen3-max
    - new-model-name  # ✅ Add here
```

### 2) Run again (existing results are preserved)

```bash
python games/evaluation/leaderboard/run_arena.py \
  --game avalon \
  --config games/games/avalon/configs/arena_config.yaml \
  --num-games 100
```



## CLI Options 🧰

| Option                | Short | Default                                                | Description                                  |
| :-------------------- | :---: | :----------------------------------------------------- | -------------------------------------------- |
| `--game`              | `-g`  | *required*                                             | Game name (`avalon` / `diplomacy`)           |
| `--config`            | `-c`  | *required*                                             | Path to arena config YAML                    |
| `--num-games`         | `-n`  | `200`                                                  | Number of games to run                       |
| `--max-workers`       | `-w`  | `10`                                                   | Max parallel workers                         |
| `--experiment-name`   |       | `arena_leaderboard_{game}`                             | Experiment name used for logs                |
| `--leaderboard-db`    |       | `games/evaluation/leaderboard/leaderboard_{game}.json` | Path to JSON DB                              |
| `--api-call-interval` |       | `0.0`                                                  | Seconds between API calls (`0.0` = no limit) |



## Rate Limiting ⏱️

To reduce API rate-limit errors under high concurrency:

```bash
--api-call-interval 1.2
```

**Suggested intervals** (example: `qwen3-max`, RPM=600):

- **5 workers**: `0.6–0.8s`
- **10 workers**: `1.0–1.2s`
- **20 workers**: `2.0–2.4s`



## Configuration ⚙️

Configs inherit from `default_config.yaml` via **Hydra**.

### Arena

```yaml
arena:
  models: [qwen-plus, qwen3-max, ...]  # Models to evaluate
  seed: 42                            # Random seed (offset by game_id)
  elo_initial: 1500                   # Initial rating value (used internally)
  elo_k: 32                           # Update rate (used internally)
```

### Game

```yaml
game:
  name: avalon                        # or diplomacy
  num_players: 5                      # Avalon: 5, Diplomacy: 7
  language: en                        # en or zh
  log_dir: games/logs/arena

  # Diplomacy-only options:
  # power_names: [AUSTRIA, ENGLAND, ...]
  # max_phases: 20
  # negotiation_rounds: 3
```

### Default Role

```yaml
default_role:
  trainable: false
  act_by_user: false
  model:
    url:  # From OPENAI_BASE_URL
    temperature: 0.7
    max_tokens: 2048
  agent:
    type: ThinkingReActAgent
    kwargs: {}  # Diplomacy: add memory config here
```

See:

- `games/games/avalon/configs/arena_config.yaml`
- `games/games/diplomacy/configs/arena_config.yaml`



## Leaderboard Data 📦

**DB location** (default):

```
games/evaluation/leaderboard/leaderboard_{game_name}.json
```

**Stored contents**:

- Per-model stats: total games, wins, role-specific win rates, and internal rating fields
- Game history (with timestamps)
- Arena configuration snapshot
- Balance metrics (computed on demand)

**Behavior**:

- ✅ Thread-safe incremental updates
- ✅ Auto-load on startup
- ✅ Add models mid-run
- ✅ Resume safely after interruptions



## Output 🧾

The displayed summary focuses on:

- 📊 **Win rate** and **total games** per model
- 🎭 **Role-specific win rates** (e.g., Merlin / Servant in Avalon)
- ⚖️ **Assignment balance** (warnings if selection ratio `< 0.8`)
- `*` Marker for insufficient games (e.g., `< 80%` of the max games among models)
