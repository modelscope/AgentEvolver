# from best_logger import print_dict
import subprocess
import argparse
import shutil
import time
import sys
import os
import signal
import shlex
from dotenv import load_dotenv
from agentevolver.utils.daemon import LaunchCommandWhenAbsent

load_dotenv()
BACK_TARGETS = os.environ.get('BACK_TARGETS', './config,./agentevolver').split(',')


def _replace_placeholder_in_config(config_obj, placeholder: str, replacement: str):
    """Recursively replace placeholder in all string values within dict/list structures.

    - Traverses dicts and lists deeply
    - Replaces all occurrences of `placeholder` inside string values
    - Leaves non-string scalars untouched
    """

    def _walk(node):
        if isinstance(node, dict):
            return {k: _walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_walk(v) for v in node]
        if isinstance(node, str):
            return node.replace(placeholder, replacement)
        return node

    return _walk(config_obj)

def parse_args():
    parser = argparse.ArgumentParser(description='The launcher of agentevolver.')
    parser.add_argument(
        '--target',
        type=str,
        default='agentevolver.main_ppo',
        required=False,
        help='Target script to run (default: agentevolver.main_ppo)'
    )
    parser.add_argument('--conf',
        type=str,
        default="",
        required=False,
        help='Path to configuration file'
    )
    parser.add_argument('--db',
        type=str,
        default="",
        required=False,
        help='Path to configuration file'
    )
    parser.add_argument('--with-appworld',
        action='store_true',  # Changed from store_true to action='store_true'
        default=False,
        help='Launch appworld'
    )
    parser.add_argument('--with-webshop',
        action='store_true',  # Changed from store_true to action='store_true'
        default=False,
        help='Launch webshop'
    )
    parser.add_argument('--with-bfcl',
        action='store_true',
        default=False,
        help='Launch bfcl'
    )
    parser.add_argument('--with-reme',
        action='store_true',
        default=False,
        help='Launch ReMe'
    )
    parser.add_argument('--with-logview',
        action='store_true',  # Changed from store_true to action='store_true'
        default=False,
        help='Launch logview'
    )
    parser.add_argument('--with-crafters',
        action='store_true',  # Changed from store_true to action='store_true'
        default=False,
        help='Launch Crafters Env Simulation'
    )
    parser.add_argument('--reboot',
        action='store_true',  # Changed from store_true to action='store_true'
        default=False,
        help='reboot flag'
    )
    parser.add_argument('-k', '--kill', '--python-killer',
        dest='python_killer',
        action='store_true',
        default=False,
        help='Kill existing ray and python processes (excluding vscode and current process) before starting')

    return parser.parse_args()


def pty_launch(service_name: str, success_std_string="Starting server on"):
    service_path = os.environ.get(f'{service_name.upper()}_PATH')
    service_script = os.environ.get(f'{service_name.upper()}_SCRIPT')
    companion = LaunchCommandWhenAbsent(
        full_argument_list=[service_script],
        dir=service_path,
        tag="appworld_env_service",
        use_pty=True
    )
    companion.launch(
        launch_wait_time=1800,
        success_std_string=success_std_string,
    )

def _fast_kill_by_keyword_bash(keyword: str, exclude_substrings=None, grace_seconds: float = 1.0):
    """Use bash pipelines to kill processes matching keyword quickly.

    - Filters out processes containing any exclude_substrings
    - Excludes current launcher process
    - Sends TERM once to all PIDs, then KILL once to all PIDs after a short grace period
    - Returns list of PIDs targeted
    """
    if exclude_substrings is None:
        exclude_substrings = ["vscode"]

    self_pid = os.getpid()

    # Build a fast PID collector using pgrep if available; fallback to ps/grep
    # We prefer pgrep -af to filter by full command and then extract PID (column 1)
    exclude_filters = " ".join([f"| grep -v -F {shlex.quote(s)}" for s in exclude_substrings])
    pid_list_cmd = (
        f"(pgrep -af -- {shlex.quote(keyword)} 2>/dev/null || true) "
        f"{exclude_filters} | awk '{{print $1}}' | grep -v -x {self_pid} || true"
    )

    try:
        res = subprocess.run(["bash", "-lc", pid_list_cmd], capture_output=True, text=True, check=False)
        pids = [pid for pid in res.stdout.split() if pid.isdigit()]
    except Exception as e:
        print(f"Failed to list PIDs via bash: {e}")
        pids = []

    # Fallback to ps/grep if pgrep path     produced nothing (e.g., no pgrep installed)
    if not pids:
        ps_pid_cmd = (
            f"ps -eo pid,command -ww | grep -F -- {shlex.quote(keyword)} | grep -v grep "
            f"{exclude_filters} | awk '{{print $1}}' | grep -v -x {self_pid} || true"
        )
        try:
            res2 = subprocess.run(["bash", "-lc", ps_pid_cmd], capture_output=True, text=True, check=False)
            pids = [pid for pid in res2.stdout.split() if pid.isdigit()]
        except Exception as e:
            print(f"Failed to list PIDs via ps/grep: {e}")
            pids = []

    if not pids:
        return []

    pid_args = " ".join(pids)
    try:
        # Send TERM to all in one call
        subprocess.run(["bash", "-lc", f"kill -TERM -- {pid_args} 2>/dev/null || true"], check=False)
        time.sleep(grace_seconds)
        # Escalate with KILL once; ignore failures for already-exited PIDs
        subprocess.run(["bash", "-lc", f"kill -KILL -- {pid_args} 2>/dev/null || true"], check=False)
    except Exception as e:
        print(f"Error issuing kill commands: {e}")

    return [int(p) for p in pids]

def _kill_processes_by_keyword(keyword: str, exclude_substrings=None, grace_seconds: float = 1.0):
    """Kill processes whose command line contains the given keyword.

    - Excludes commands containing any of exclude_substrings
    - Never kills the current process
    - Sends SIGTERM first, then SIGKILL if the process is still alive after grace period
    """
    if exclude_substrings is None:
        exclude_substrings = ["vscode"]

    try:
        # Use ps to get PID and full command
        ps = subprocess.run(["ps", "-eo", "pid,command", "-ww"], capture_output=True, text=True, check=True)
        lines = ps.stdout.strip().splitlines()
    except Exception as e:
        print(f"Failed to list processes: {e}")
        return []

    killed = []
    this_pid = os.getpid()

    # Skip header if present
    for line in lines[1:] if lines and lines[0].lower().startswith("pid") else lines:
        line = line.strip()
        if not line:
            continue
        try:
            pid_str, cmd = line.split(None, 1)
            pid = int(pid_str)
        except ValueError:
            continue

        # Filters
        if pid == this_pid:
            continue
        if keyword not in cmd:
            continue
        if any(ex in cmd for ex in exclude_substrings):
            continue

        # Try SIGTERM then SIGKILL
        try:
            os.kill(pid, signal.SIGTERM)
            # brief wait for graceful shutdown
            deadline = time.time() + grace_seconds
            while time.time() < deadline:
                try:
                    os.kill(pid, 0)
                except OSError:
                    # process gone
                    break
                time.sleep(0.1)
            else:
                # still alive -> SIGKILL
                os.kill(pid, signal.SIGKILL)
            killed.append(pid)
            print(f"Killed PID {pid} matching '{keyword}'")
        except ProcessLookupError:
            # already gone
            continue
        except PermissionError:
            print(f"No permission to kill PID {pid}; skipping")
        except Exception as e:
            print(f"Error killing PID {pid}: {e}")

    return killed

def main():
    args = parse_args()

    # Optionally kill existing processes before starting
    if getattr(args, 'python_killer', False):
        print("--python-killer enabled: fast killing via bash for 'ray' and 'python' (excluding 'vscode' and current process)")
        killed_ray = _fast_kill_by_keyword_bash("ray", exclude_substrings=["vscode"], grace_seconds=0.8)
        killed_py = _fast_kill_by_keyword_bash("python", exclude_substrings=["vscode"], grace_seconds=0.8)
        print(f"Targeted {len(killed_ray)} ray-related processes and {len(killed_py)} python processes.")
        # small pause to allow system to release resources (ports, files)
        time.sleep(0.5)

    if args.conf:
        yaml_path = args.conf
        assert yaml_path.endswith('.yaml'), "Configuration file must be a YAML file"
        exp_base = os.path.dirname(args.conf)

        if os.path.exists(exp_base):

            ## 0. read yaml (get trainer.experiment_name)
            import yaml
            with open(yaml_path, 'r') as file:
                config = yaml.safe_load(file)
            exp_name = config.get('trainer').get('experiment_name')
            if exp_name is None or exp_name == 'read_yaml_name':
                if exp_name is not None: exp_name = exp_name.replace('|', '-')
                exp_name = os.path.basename(yaml_path).replace('.yaml', '')
            else:
                exp_name = exp_name.replace('|', '-')

            print('----------------------------------------')
            backup_dir = os.path.join('launcher_record', exp_name, 'backup')
            yaml_backup_dst = os.path.join('launcher_record', exp_name, 'yaml_backup.yaml')
            exe_yaml_path = yaml_backup_dst
            exe_exp_base = os.path.dirname(yaml_backup_dst)
            print('Experiment Name:', exp_name)
            print('Experiment Backup Dir:', backup_dir)
            print('Experiment Yaml Dir:', yaml_backup_dst)
            print('----------------------------------------')
            time.sleep(2)

            ## 1. check exp_base/backup exist
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
            else:
                total_seconds = 10
                for i in range(total_seconds):
                    print(f"\rWarning: backup directory already exists, we will automatically ignore this after {total_seconds - i} seconds...", end="", flush=True)
                    time.sleep(1)

            ## 2. copy files to backup
            for backup_target in BACK_TARGETS:
                print(f"Copying {backup_target} to {os.path.join(backup_dir, os.path.basename(backup_target))}")
                shutil.copytree(backup_target, os.path.join(backup_dir, os.path.basename(backup_target)), dirs_exist_ok=True)

            ## 3. copy yaml to backup
            yaml_backup_src = yaml_path
            shutil.copyfile(yaml_backup_src, yaml_backup_dst)

            ## 4. edit new yaml
            yaml_path = yaml_backup_dst
            # now, replace the trainer.experiment_name
            with open(yaml_path, 'r') as file:
                config = yaml.safe_load(file)
            config['trainer']['experiment_name'] = exp_name
            # scan all config item in config recursively, find string "${trainer.experiment_name}", replace with `exp_name`
            config = _replace_placeholder_in_config(
                config,
                placeholder="${trainer.experiment_name}",
                replacement=exp_name,
            )

            # replace all
            with open(yaml_path, 'w') as file:
                yaml.dump(config, file)

        else:
            raise FileNotFoundError(f"Configuration file not found: {exp_base}")

        env = os.environ.copy()
        if args.db:
            env["RAY_DEBUG_POST_MORTEM"] = "1"
            env["DEBUG_TAGS"] = args.db
            env["RAY_record_task_actor_creation_sites"] =  "true"
            print("Debug mode is ON")
        else:
            print("Debug mode is OFF")

    if args.with_reme:
        # test done
        pty_launch("reme", success_std_string="Uvicorn running on")

    if args.with_appworld:
        # test done
        pty_launch("appworld")

    if args.with_crafters:
        # test done
        pty_launch("crafters")

    if args.with_webshop:
        # not tesed
        pty_launch("webshop")

    if args.with_bfcl:
        pty_launch("bfcl")

    if args.with_logview:

        companion = LaunchCommandWhenAbsent(
            full_argument_list=[
                sys.executable,
                '-m',
                'web_display.start_web',
            ],
            dir='./',
            tag="logview"
        )
        companion.launch(launch_wait_time=1800,success_std_string="Uvicorn running on", env_dict={})
        time.sleep(2.5)
        try:
            import webbrowser
            from datetime import datetime
            final_log_path = os.path.join( "experiments", exp_name, "trace_rollout", datetime.now().strftime("%Y_%m_%d_%H_%M"))
            # make dir
            os.makedirs(final_log_path)
            webbrowser.open("http://127.0.0.1:8181/"+"?path="+os.path.abspath(final_log_path))
        except:
            pass

    if args.conf:
        # let's begin the training process
        cmd = [
            sys.executable,
            '-m',
            args.target,
            '--config-path',
            os.path.abspath(exe_exp_base),
            '--config-name',
            os.path.basename(exe_yaml_path),
        ]

        if args.with_logview:
            env.update({
                'BEST_LOGGER_WEB_SERVICE_URL': os.environ.get('BEST_LOGGER_WEB_SERVICE_URL', 'http://127.0.0.1:8181/')
            })

        try:
            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, cwd=os.path.abspath('./'), env=env)
        except subprocess.CalledProcessError as e:
            print(f"Error running subprocess: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()