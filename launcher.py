import subprocess
import argparse
import shutil
import time
import sys
import os

BACK_TARGETS = [
    './config',
    './beyondagent',
]


def _default_envservice_root() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'external', 'EnvService')
    )


def _launch_env_service_daemon(
    *,
    python_exe: str,
    env_name: str,
    portal: str,
    port: int,
    extra_env: dict,
) -> None:
    """Same as the exec line in bfcl.sh / appworld.sh: run env_service under ENVSERVICE_ROOT."""
    from beyondagent.rollout_with_daemon import LaunchCommandWhenAbsent

    env_service_root = os.environ.get('ENVSERVICE_ROOT') or _default_envservice_root()
    if not os.path.isdir(env_service_root):
        raise RuntimeError(
            f"ENVSERVICE_ROOT does not exist: {env_service_root}\n"
            "Set ENVSERVICE_ROOT or ensure SeeUPO/external/EnvService is available."
        )

    env = os.environ.copy()
    env.update(extra_env)
    # Same as in launch scripts: PYTHONPATH includes the EnvService root
    pp = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = (
        env_service_root if not pp else env_service_root + os.pathsep + pp
    )

    companion = LaunchCommandWhenAbsent(
        full_argument_list=[
            python_exe,
            '-m',
            'env_service.env_service',
            '--env',
            env_name,
            '--portal',
            portal,
            '--port',
            str(port),
        ],
        dir=env_service_root,
        tag=f'{env_name}_env_service',
    )
    companion.launch(
        launch_wait_time=1800,
        success_std_string='Uvicorn running on',
        env_dict=env,
    )


def _bfcl_default_env_vars(env_service_root: str) -> dict:
    """Default paths aligned with bfcl.sh (used when not manually exported)."""
    env_service_dir = os.path.join(env_service_root, 'env_service')
    default_bfcl_dir = os.path.join(env_service_dir, 'environments', 'bfcl')
    bfcl_env_dir = os.environ.get('BFCL_ENV_DIR', default_bfcl_dir)
    return {
        'ENV_PATH': bfcl_env_dir,
        'BFCL_DATA_PATH': os.path.join(
            bfcl_env_dir, 'bfcl_data', 'multi_turn_base_processed.jsonl'
        ),
        'BFCL_SPLID_ID_PATH': os.path.join(
            bfcl_env_dir, 'bfcl_data', 'multi_turn_base_split_ids.json'
        ),
        'BFCL_ANSWER_PATH': os.path.join(bfcl_env_dir, 'bfcl_eval', 'possible_answer'),
        'RAY_ENV_NAME': 'bfcl',
    }


def _appworld_default_env_vars(env_service_root: str) -> dict:
    """Align APPWORLD_ROOT / RAY_ENV_NAME with appworld.sh."""
    env_service_dir = os.path.join(env_service_root, 'env_service')
    default_root = os.path.join(env_service_dir, 'environments', 'appworld')
    return {
        'APPWORLD_ROOT': os.environ.get('APPWORLD_ROOT', default_root),
        'NODE_ENV': os.environ.get('NODE_ENV', 'production'),
        'RAY_ENV_NAME': 'appworld',
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='BA Launcher (unified EnvService BFCL/AppWorld logic)'
    )
    parser.add_argument(
        '--conf',
        type=str,
        required=True,
        help='Path to configuration file',
    )
    parser.add_argument(
        '--db',
        type=str,
        default='',
        required=False,
        help='Path to configuration file',
    )
    parser.add_argument(
        '--with-appworld',
        action='store_true',
        default=False,
        help='Launch AppWorld via bundled EnvService (same pattern as BFCL)',
    )
    parser.add_argument(
        '--with-bfcl',
        action='store_true',
        default=False,
        help='Launch BFCL via bundled EnvService (same pattern as AppWorld)',
    )
    parser.add_argument(
        '--with-webshop',
        action='store_true',
        default=False,
        help='Launch webshop',
    )

    return parser.parse_args()


def main():
    args = parse_args()
    yaml_path = args.conf
    assert yaml_path.endswith('.yaml'), 'Configuration file must be a YAML file'
    exp_base = os.path.dirname(args.conf)

    if os.path.exists(exp_base):

        ## 0. read yaml (get trainer.experiment_name)
        import yaml

        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        exp_name = config.get('trainer').get('experiment_name')
        exp_name = exp_name.replace('|', '-')

        ## 1. check exp_base/backup exist
        backup_dir = os.path.join(exp_base, exp_name, 'backup')
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
        else:
            total_time = 5
            for i in range(total_time):
                print(
                    f'warning: backup directory already exists, we will automatically ignore this after {total_time - i} seconds...'
                )
                time.sleep(1)

        ## 2. copy files to backup
        for backup_target in BACK_TARGETS:
            print(
                f'Copying {backup_target} to {os.path.join(backup_dir, os.path.basename(backup_target))}'
            )
            shutil.copytree(
                backup_target,
                os.path.join(backup_dir, os.path.basename(backup_target)),
                dirs_exist_ok=True,
            )

        ## 3. copy yaml to backup
        yaml_backup_src = yaml_path
        yaml_backup_dst = os.path.join(exp_base, exp_name, 'yaml_backup.yaml')
        shutil.copyfile(yaml_backup_src, yaml_backup_dst)

    else:
        raise FileNotFoundError(f'Configuration file not found: {exp_base}')

    env = os.environ.copy()
    if args.db:
        env['RAY_DEBUG_POST_MORTEM'] = '1'
        env['DEBUG_TAGS'] = args.db
        env['RAY_record_task_actor_creation_sites'] = 'true'
        print('Debug mode is ON')
    else:
        print('Debug mode is OFF')

    env_service_root = os.environ.get('ENVSERVICE_ROOT') or _default_envservice_root()

    if args.with_bfcl:
        bfcl_python = os.environ.get('BFCL_PYTHON')
        if not bfcl_python:
            raise RuntimeError(
                'Set BFCL_PYTHON to the Python executable that has BFCL dependencies installed\n'
                '(same environment as conda activate envservice_bfcl in qwen3-ppo-bfcl.sh).'
            )
        portal = os.environ.get('BFCL_PORTAL', '127.0.0.1')
        port = int(os.environ.get('BFCL_PORT', '8080'))
        extra = _bfcl_default_env_vars(env_service_root)
        _launch_env_service_daemon(
            python_exe=bfcl_python,
            env_name='bfcl',
            portal=portal,
            port=port,
            extra_env=extra,
        )

    if args.with_appworld:
        appworld_python = os.environ.get('APPWORLD_PYTHON')
        if not appworld_python:
            raise RuntimeError(
                'Set APPWORLD_PYTHON to the Python executable with appworld installed\n'
                '(same conda env as after running env_service/environments/appworld/setup.sh).'
            )
        portal = os.environ.get('APPWORLD_PORTAL', '127.0.0.1')
        port = int(os.environ.get('APPWORLD_PORT', '8000'))
        extra = _appworld_default_env_vars(env_service_root)
        work_root = os.environ.get('WORKSPACE_DIR')
        if work_root:
            extra['WORKSPACE_DIR'] = work_root
        _launch_env_service_daemon(
            python_exe=appworld_python,
            env_name='appworld',
            portal=portal,
            port=port,
            extra_env=extra,
        )

    # if args.with_webshop:
    #     from beyondagent.rollout_with_daemon import LaunchCommandWhenAbsent
    #     if os.path.exists('<INTERNAL_ENV_SERVICE_PATH>/EnvService'):

    # let's begin the training process
    cmd = [
        sys.executable,
        '-m',
        'beyondagent.main_ppo',
        '--config-path',
        os.path.abspath(exp_base),
        '--config-name',
        os.path.basename(yaml_path),
    ]

    try:
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=os.path.abspath('./'), env=env)
    except subprocess.CalledProcessError as e:
        print(f'Error running subprocess: {e}')
        sys.exit(1)
    except Exception as e:
        print(f'Unexpected error: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
