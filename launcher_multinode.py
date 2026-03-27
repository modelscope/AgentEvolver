#!/usr/bin/env python3
"""
Fully compatible with single-node and multi-node training.
"""
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


def parse_args():
    parser = argparse.ArgumentParser(description='BA Multi-Node Launcher')
    parser.add_argument('--conf', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--db', type=str, default="", required=False, help='Debug tags')
    parser.add_argument('--with-bfcl', action='store_true', default=False, help='Check BFCL service')
    parser.add_argument('--with-appworld', action='store_true', default=False, help='Check AppWorld service')
    return parser.parse_args()


def get_distributed_info():
    """Collect distributed training metadata from the environment."""
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '29500')
    
    is_distributed = world_size > 1
    is_main_node = rank == 0
    
    return {
        'rank': rank,
        'world_size': world_size,
        'local_rank': local_rank,
        'master_addr': master_addr,
        'master_port': master_port,
        'is_distributed': is_distributed,
        'is_main_node': is_main_node
    }


def print_distributed_info(dist_info):
    """Print a short summary of distributed setup."""
    print("=" * 70)
    if dist_info['is_distributed']:
        print("🚀 Multi-Node Distributed Training")
        print(f"   Rank: {dist_info['rank']}/{dist_info['world_size']}")
        print(f"   Master: {dist_info['master_addr']}:{dist_info['master_port']}")
        print(f"   Role: {'🎯 Main Node' if dist_info['is_main_node'] else '⚙️ Worker Node'}")
    else:
        print("🖥️  Single-Node Training")
    print("=" * 70)


def verify_checkpoint_config(yaml_path, dist_info):
    """Verify checkpoint-related settings in the YAML config."""
    import yaml
    
    print(f"\n[Node {dist_info['rank']}] Verifying checkpoint configuration...")
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    trainer_config = config.get('trainer', {})
    
    # Check key trainer fields
    default_local_dir = trainer_config.get('default_local_dir')
    default_hdfs_dir = trainer_config.get('default_hdfs_dir')
    exp_name = trainer_config.get('experiment_name', 'unknown')
    save_freq = trainer_config.get('save_freq', 'not set')
    
    print(f"📁 Checkpoint Configuration:")
    print(f"   Experiment name: {exp_name}")
    print(f"   default_local_dir: {default_local_dir}")
    print(f"   default_hdfs_dir: {default_hdfs_dir}")
    print(f"   save_freq: {save_freq}")
    
    if default_local_dir is None:
        print("   ⚠️  WARNING: default_local_dir not set in YAML!")
        print("   Checkpoint may be saved to unexpected location!")
        return False
    
    # Ensure default_local_dir is absolute
    if not os.path.isabs(default_local_dir):
        print(f"   ⚠️  WARNING: default_local_dir is relative path: {default_local_dir}")
        print(f"   Will be resolved relative to: {os.getcwd()}")
        # Resolve relative path to absolute
        default_local_dir = os.path.abspath(default_local_dir)
        print(f"   Resolved to: {default_local_dir}")
    
    # Ensure directory exists and is writable (main node creates if missing)
    if dist_info['is_main_node']:
        if not os.path.exists(default_local_dir):
            print(f"   📂 Creating checkpoint directory: {default_local_dir}")
            try:
                os.makedirs(default_local_dir, exist_ok=True)
                print(f"   ✅ Directory created successfully")
            except Exception as e:
                print(f"   ❌ Failed to create directory: {e}")
                return False
        
        # Test write permission
        test_file = os.path.join(default_local_dir, '.write_test_main')
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            print(f"   ✅ Write permission verified")
        except Exception as e:
            print(f"   ❌ Write permission test failed: {e}")
            return False
    else:
        # Workers only check that the path is reachable
        if os.path.exists(default_local_dir):
            print(f"   ✅ Directory accessible from worker node")
        else:
            print(f"   ⚠️  Directory not accessible from worker node")
            print(f"   This may cause issues if not using shared filesystem!")
    
    print("=" * 70)
    return True


def setup_experiment_backup(yaml_path, dist_info):
    """Set up experiment backup (file copy runs on main node only)."""
    exp_base = os.path.dirname(yaml_path)
    
    import yaml
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    exp_name = config.get('trainer', {}).get('experiment_name', 'default_exp')
    exp_name = exp_name.replace('|', '-')
    
    # All nodes ensure the backup directory exists
    backup_dir = os.path.join(exp_base, exp_name, 'backup')
    if not os.path.exists(backup_dir):
        try:
            os.makedirs(backup_dir, exist_ok=True)
            print(f"[Node {dist_info['rank']}] Created backup directory: {backup_dir}")
        except Exception as e:
            print(f"[Node {dist_info['rank']}] Warning: Failed to create backup dir: {e}")
    
    if dist_info['is_distributed'] and not dist_info['is_main_node']:
        print(f"[Node {dist_info['rank']}] Skipping backup files (worker node)")
        return exp_base, exp_name
    
    # Main node performs the backup copy
    print(f"[Node {dist_info['rank']}] Performing experiment backup...")
    
    for backup_target in BACK_TARGETS:
        if os.path.exists(backup_target):
            target_path = os.path.join(backup_dir, os.path.basename(backup_target))
            print(f"📦 Backing up {backup_target} → {target_path}")
            try:
                shutil.copytree(backup_target, target_path, dirs_exist_ok=True)
            except Exception as e:
                print(f"⚠️  Warning: Failed to backup {backup_target}: {e}")
    
    yaml_backup_dst = os.path.join(exp_base, exp_name, 'yaml_backup.yaml')
    try:
        shutil.copyfile(yaml_path, yaml_backup_dst)
        print(f"✅ Config backed up to {yaml_backup_dst}")
    except Exception as e:
        print(f"⚠️  Warning: Failed to backup YAML: {e}")
    
    return exp_base, exp_name


def check_bfcl_service(dist_info):
    """Probe BFCL HTTP health (main node only)."""
    if not dist_info['is_main_node']:
        print(f"[Node {dist_info['rank']}] Skipping BFCL check (worker node)")
        return
    
    print(f"[Node {dist_info['rank']}] Checking BFCL service...")
    
    import requests
    service_url = os.environ.get('BFCL_SERVICE_URL', 'http://localhost:8080')
    max_wait = 120
    start_time = time.time()
    
    print(f"🔍 Probing BFCL service at {service_url}...")
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(f"{service_url}/health", timeout=3)
            if response.status_code == 200:
                print(f"✅ BFCL service ready at {service_url}")
                print(f"   Response time: {response.elapsed.total_seconds():.2f}s")
                return
        except requests.exceptions.ConnectionError:
            pass
        except Exception as e:
            print(f"   Connection attempt failed: {type(e).__name__}")
        
        elapsed = int(time.time() - start_time)
        if elapsed % 10 == 0:
            remaining = max_wait - elapsed
            print(f"⏳ Waiting for BFCL... ({elapsed}s elapsed, {remaining}s remaining)")
        time.sleep(2)
    
    raise RuntimeError(
        f"❌ BFCL service not available at {service_url} after {max_wait}s\n"
        f"   Please check BFCL service logs and ensure it's running."
    )


def check_appworld_service(dist_info):
    """Probe AppWorld HTTP health (main node only)."""
    if not dist_info['is_main_node']:
        print(f"[Node {dist_info['rank']}] Skipping AppWorld check (worker node)")
        return
    
    print(f"[Node {dist_info['rank']}] Checking AppWorld service...")
    
    import requests
    service_url = os.environ.get('APPWORLD_SERVICE_URL', 'http://localhost:8000')
    max_wait = 120
    start_time = time.time()
    
    print(f"🔍 Probing AppWorld service at {service_url}...")
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(f"{service_url}/health", timeout=3)
            if response.status_code == 200:
                print(f"✅ AppWorld service ready at {service_url}")
                return
        except:
            pass
        
        elapsed = int(time.time() - start_time)
        if elapsed % 10 == 0:
            print(f"⏳ Waiting for AppWorld... ({elapsed}s)")
        time.sleep(2)
    
    raise RuntimeError(f"❌ AppWorld service not available at {service_url} after {max_wait}s")


def verify_ray_cluster(dist_info):
    """Verify Ray cluster connectivity and print a short status."""
    print(f"[Node {dist_info['rank']}] Verifying Ray cluster...")
    
    if not dist_info['is_distributed']:
        print("Single-node mode: Ray will be initialized by veRL")
        return
    
    try:
        import ray
        ray_address = f"{dist_info['master_addr']}:6379"
        
        if not ray.is_initialized():
            print(f"Connecting to Ray cluster at {ray_address}")
            ray.init(address=ray_address, ignore_reinit_error=True)
        
        # Fetch cluster view
        nodes = ray.nodes()
        resources = ray.cluster_resources()
        expected_nodes = dist_info['world_size']
        
        print("\n" + "=" * 70)
        print(f"🎯 Ray Cluster Status (Node {dist_info['rank']})")
        print("=" * 70)
        print(f"  Total Nodes: {len(nodes)}/{expected_nodes}")
        print(f"  Total GPUs:  {int(resources.get('GPU', 0))}")
        print(f"  Total CPUs:  {int(resources.get('CPU', 0))}")
        print(f"  Memory:      {resources.get('memory', 0) / 1e9:.1f} GB")
        
        if dist_info['is_main_node']:
            print("\n  Node Details:")
            for i, node in enumerate(nodes):
                node_resources = node.get('Resources', {})
                print(f"    Node {i}: {int(node_resources.get('GPU', 0))} GPUs, "
                      f"{int(node_resources.get('CPU', 0))} CPUs")
        
        print("=" * 70 + "\n")
        
        if len(nodes) < expected_nodes:
            print(f"⚠️  Warning: Only {len(nodes)}/{expected_nodes} nodes in cluster")
        
        # Shut down; veRL will initialize Ray again
        ray.shutdown()
        
    except Exception as e:
        print(f"⚠️  Warning: Could not verify Ray cluster: {e}")
        print("Continuing anyway - veRL will handle Ray initialization")


def main():
    args = parse_args()
    yaml_path = args.conf
    
    # 1. Distributed environment
    dist_info = get_distributed_info()
    print_distributed_info(dist_info)
    
    # 2. Checkpoint config (critical)
    print("\n" + "=" * 70)
    print("🔍 Step 1: Verifying Checkpoint Configuration")
    print("=" * 70)
    if not verify_checkpoint_config(yaml_path, dist_info):
        print("❌ Checkpoint configuration verification failed!")
        print("Please ensure 'default_local_dir' is set in your YAML config.")
        sys.exit(1)
    
    # 3. Experiment backup (copy on main node only)
    print("\n" + "=" * 70)
    print(f"🔍 Step 2: Setting Up Experiment Backup")
    print("=" * 70)
    exp_base, exp_name = setup_experiment_backup(yaml_path, dist_info)
    
    # 4. Debug-related env vars
    env = os.environ.copy()
    if args.db:
        env["RAY_DEBUG_POST_MORTEM"] = "1"
        env["DEBUG_TAGS"] = args.db
        env["RAY_record_task_actor_creation_sites"] = "true"
        print(f"\n[Node {dist_info['rank']}] 🐛 Debug mode is ON (tags: {args.db})")
    else:
        print(f"\n[Node {dist_info['rank']}] Debug mode is OFF")
    
    # 5. Optional env-service health checks (main node only)
    if args.with_bfcl:
        print("\n" + "=" * 70)
        print(f"🔍 Step 3: Checking BFCL Service")
        print("=" * 70)
        check_bfcl_service(dist_info)
    
    if args.with_appworld:
        print("\n" + "=" * 70)
        print(f"🔍 Step 4: Checking AppWorld Service")
        print("=" * 70)
        check_appworld_service(dist_info)
    
    # 6. Ray cluster (all nodes)
    print("\n" + "=" * 70)
    print(f"🔍 Step 5: Verifying Ray Cluster")
    print("=" * 70)
    verify_ray_cluster(dist_info)
    
    # 7. Print key environment variables
    print("\n" + "=" * 70)
    print("🔍 Environment Check")
    print("=" * 70)
    print(f"   RAY_ADDRESS: {env.get('RAY_ADDRESS', 'Not set')}")
    print(f"   PYTHONPATH: {env.get('PYTHONPATH', 'Not set')[:100]}...")
    print(f"   PROJECT_PATH: {env.get('PROJECT_PATH', 'Not set')}")
    print(f"   CHECKPOINT_BASE: {env.get('CHECKPOINT_BASE', 'Not set')}")
    print(f"   Current directory: {os.getcwd()}")
    print("=" * 70)
    
    # 8. Start training (each node runs this once)
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
        print(f"\n[Node {dist_info['rank']}] 🚀 Starting training process...")
        print(f"Command: {' '.join(cmd)}")
        print(f"Working directory: {os.path.abspath('./')}")
        print("=" * 70 + "\n")
        
        subprocess.run(cmd, check=True, cwd=os.path.abspath('./'), env=env)
        
        print(f"\n[Node {dist_info['rank']}] ✅ Training completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ [Node {dist_info['rank']}] Error running subprocess: {e}")
        print(f"Return code: {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n[Node {dist_info['rank']}] ⚠️  Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ [Node {dist_info['rank']}] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()