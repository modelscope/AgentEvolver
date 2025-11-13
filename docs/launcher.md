# Launcher Usage Guide (`launcher.py`)

This document explains the usage, arguments, and workflow of `launcher.py`, the main entry point for launching and managing experiments in the AgentEvolver project.

## Overview

`launcher.py` is a command-line tool designed to:
- Prepare experiment environments
- Backup configuration and code
- Optionally kill existing Python/Ray processes
- Launch various environment services (AppWorld, WebShop, BFCL, Exp Maker, LogView, Crafters)
- Start the main training or evaluation process

## Basic Usage


Run the launcher with desired options. Example:

```bash
python launcher.py --conf examples/self-question-nav-attr.yaml --python-killer --with-appworld --with-exp-maker --with-logview
```
Explaination:
- `--conf`: Choose a experiment yaml to launch training (the primary argument).
- `--python-killer`: Kill existing all Python and Ray processes (except VSCode and itself, use with caution if you are running some other training jobs).
- `--with-appworld`: Let the launcher help you start `Appworld` environment service (persistent and can avoid re-launching).
- `--with-exp-maker`: Let the launcher help you start `ReMe` environment service (persistent and can avoid re-launching).
- `--with-logview`: Open rollout log viewer. If you are using VSCode, a browser windows will pop out automatically.


## Command-Line Arguments

| Argument                | Type      | Description                                                                                 |
|------------------------|-----------|---------------------------------------------------------------------------------------------|
| `--target`             | str       | Target script/module to run (default: `agentevolver.main_ppo`)                               |
| `--conf`               | str       | Path to the experiment YAML configuration file                                               |
| `--db`                 | str       | Enable debug mode and set debug tags                                                         |
| `--with-appworld`      | flag      | Launch AppWorld environment service                                                          |
| `--with-webshop`       | flag      | Launch WebShop environment service                                                           |
| `--with-bfcl`          | flag      | Launch BFCL environment service                                                             |
| `--with-exp-maker`     | flag      | Launch Experience Maker service                                                              |
| `--with-logview`       | flag      | Launch LogView web service and open browser                                                  |
| `--with-crafters`      | flag      | Launch Crafters environment simulation                                                      |
| `-k`, `--kill`, `--python-killer` | flag | Kill existing Ray and Python processes before starting (recommended for clean start)         |

## Workflow Details

1. **Process Cleanup (Optional)**
   - If `--python-killer` is set, kills all Ray and Python processes except VSCode and itself.

2. **Configuration Handling**
   - Loads the YAML config file specified by `--conf`.
   - Determines experiment name from `trainer.experiment_name` or YAML filename.
   - Backs up code/config directories and the YAML file to `launcher_record/<exp_name>/backup/`.
   - Rewrites the YAML to set the correct experiment name and replace placeholders.

3. **Service Launching**
   - Launches selected environment services (AppWorld, WebShop, BFCL, Exp Maker, LogView, Crafters) as background processes using PTY.
   - For LogView, opens the web UI in a browser.

4. **Main Process Launch**
   - Runs the main training/evaluation script/module with the prepared config.
   - Passes environment variables for debugging and logging as needed.


## Persistent service management and logs

Services launched with flags like `--with-appworld`, `--with-exp-maker`, etc., are started via a companion process manager built on `agentevolver.utils.daemon.LaunchCommandWhenAbsent`.

What this gives you:

- Single-instance guarantee: if a service is already running, it won’t be launched again.
- Detached background execution: services keep running independently of your terminal session.
- Stable identifiers: each launch combination is hashed; its process group id (PGID) is stored next to the log.
- Structured logs: all companion logs live under `logs/companion/` with a file name pattern `<tag>.<hash>.<hostname>.log` and a sibling PGID file `<tag>.<hash>.<hostname>.pgid`.

On launch, the console prints where logs go, for example:

```
log to logs/companion/appworld_env_service.bc21d4e3.dlc1x89r0ps6ysm8-master-0.log
```

You can open or stream the service logs directly, e.g.:

```bash
tail -f logs/companion/appworld_env_service.*.log
```

Notes:

- Skips re-launch if an existing PGID is active. To restart, stop the service (kill the process group) or remove the corresponding `.pgid` file and relaunch the flag.
- For PTY-backed services, human-readable command execution is proxied via `agentevolver.utils.pty`; output still goes to the same log file.
- Tags: `launcher.py` applies a meaningful `tag` for each service (e.g., `appworld_env_service`), which appears in the log filename.


## Example: Full Experiment Launch

```bash
python launcher.py --kill --conf examples/self-question-nav-attr.yaml --with-appworld --with-exp-maker
```

- Cleans up old processes
- Backs up code and config
- Launches AppWorld and Exp Maker
- Starts the experiment with the specified YAML config

## Notes
- The backup directory is `launcher_record/<exp_name>/backup/`.
- The rewritten YAML is saved as `launcher_record/<exp_name>/yaml_backup.yaml`.
- Placeholders like `${trainer.experiment_name}` in the config are replaced automatically.
- For debugging, use `--db <tag>` to enable debug mode and set debug tags.
- To launch additional services, add their respective flags.

## Troubleshooting
- If you see errors about missing config or backup directories, check your YAML path and permissions.
- If services fail to start, ensure their paths/scripts are set in your environment variables or `.env` file.
- For port or resource conflicts, use `--kill` to clean up before launching.

---

For more details, see the comments in `launcher.py` or contact the project maintainers.

## Debugging with `--db`

Use `--db` to enable post-mortem debugging and set tag-based conditional breakpoints. The launcher will set:

- `RAY_DEBUG_POST_MORTEM=1`
- `DEBUG_TAGS=<value of --db>` (use `|` to separate multiple tags)
- `RAY_record_task_actor_creation_sites=true`

Then, inside your code, import the helper from `vsdb.py`:

```python
from agentevolver import bp

def some_function():
   bp("tag")  # hits only if "tag" is in DEBUG_TAGS and RAY_DEBUG_POST_MORTEM is set
   # ... your logic ...
```

Run with one or more tags:

```bash
python launcher.py --kill --conf examples/self-question-nav-attr.yaml --db "tag|tag2|tag3|tag4"
```

Behavior summary:

- `bp()` with no arguments: triggers once when `RAY_DEBUG_POST_MORTEM` is set.
- `bp("tag")`: triggers (once by default) only if `tag` appears in `DEBUG_TAGS` (split by `|`).
- You can call `vscode_conditional_breakpoint(tag, once=False)` if you need to break every time.

Tip: The approach works well with the Ray Distributed Debugger VSCode extension. See the inline guide in `vsdb.py` for setup screenshots and more details.

## How to add a new environment service

You can extend the launcher with your own background service (similar to AppWorld, WebShop, etc.). The launcher already provides a small helper, `pty_launch(service_name, success_std_string)`, that reads your service settings from environment variables and starts it as a managed, single-instance background process.

Follow these three steps:

1) Add a CLI flag

- Open `launcher.py`, find `parse_args()`, and add a boolean flag for your service. Example for a service named "MyEnv":

   - `parser.add_argument('--with-myenv', action='store_true', default=False, help='Launch MyEnv service')`

2) Call `pty_launch("myenv")` in `main()`

- In `main()`, add a conditional block that calls the helper when the flag is used:

   - `if args.with_myenv: pty_launch("myenv", success_std_string="Uvicorn running on")`

- The second argument is an optional "ready" text the launcher watches for in the service output. Pick a short substring that appears when your service is fully up, e.g.:
   - "Starting server on"
   - "Uvicorn running on"
   - "Listening on"

3) Provide `.env` entries for your service

`launcher.py` loads environment variables via `python-dotenv`. Define two variables so `pty_launch()` knows how and where to start your service:

- `MYENV_PATH`   — Working directory to run the service in (usually the repo/subfolder containing scripts)
- `MYENV_SCRIPT` — The exact command used to start the service

Example `.env` snippet (place at project root):

```
# MyEnv service
MYENV_PATH=/abs/path/to/myenv
MYENV_SCRIPT=python -m myenv.api --host 0.0.0.0 --port 9009
```

Quick check:

- After the edits above, try:
   - `python launcher.py --with-myenv` (add `--kill` to clean stale processes if needed)
   - Watch logs under `logs/companion/` to confirm startup completes and the ready string appears.
