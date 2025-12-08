# -*- coding: utf-8 -*-
"""Generate training task files for Avalon game.

This script generates a JSONL file containing Task objects for training.
Each task has the same configuration (based on task_config.yaml) but different task_id.
The assassin role is set to trainable: true, other roles use default configuration.
"""
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any


def load_task_config(config_path: str) -> Dict[str, Any]:
    """Load task configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_avalon_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create avalon config with assassin set to trainable, other roles use default."""
    # Create a minimal config with only default_model and game
    avalon_config = {
        'default_model': base_config.get('default_model', {}),
        'game': base_config.get('game', {}),
        'roles': {
            'assassin': {
                'trainable': True
            }
        }
    }
    
    return avalon_config


def create_task(task_id: str, avalon_config: Dict[str, Any], env_type: str = "avalon") -> Dict[str, Any]:
    """Create a Task object as dictionary."""
    return {
        "task_id": task_id,
        "env_type": env_type,
        "open_query": False,
        "metadata": {
            "avalon_config": avalon_config
        },
        "query": None,
        "ground_truth": None,
        "evaluator": "env"
    }


def generate_tasks(
    config_path: str,
    output_path: str,
    num_tasks: int = 10,
    env_type: str = "avalon",
    task_id_prefix: str = "avalon_train"
):
    """Generate training task file.
    
    Args:
        config_path: Path to task_config.yaml
        output_path: Path to output JSONL file
        num_tasks: Number of tasks to generate
        env_type: Environment type for tasks
        task_id_prefix: Prefix for task IDs
    """
    # Load base configuration
    base_config = load_task_config(config_path)
    
    # Create avalon config with assassin trainable
    avalon_config = create_avalon_config(base_config)
    
    # Generate tasks
    tasks = []
    for i in range(num_tasks):
        task_id = f"{task_id_prefix}_{i:04d}"
        task = create_task(task_id, avalon_config, env_type)
        tasks.append(task)
    
    # Write to JSONL file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + '\n')
    
    print(f"Generated {num_tasks} tasks and saved to {output_path}")
    print(f"Assassin role is set to trainable: true")
    print(f"Other roles use default configuration")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training task files for Avalon game")
    parser.add_argument(
        "--config",
        type=str,
        default="games/avalon/task_config.yaml",
        help="Path to task_config.yaml"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="games/avalon/train_tasks.jsonl",
        help="Path to output JSONL file"
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=10,
        help="Number of tasks to generate"
    )
    parser.add_argument(
        "--env_type",
        type=str,
        default="avalon",
        help="Environment type for tasks"
    )
    parser.add_argument(
        "--task_id_prefix",
        type=str,
        default="avalon_train",
        help="Prefix for task IDs"
    )
    
    args = parser.parse_args()
    
    generate_tasks(
        config_path=args.config,
        output_path=args.output,
        num_tasks=args.num_tasks,
        env_type=args.env_type,
        task_id_prefix=args.task_id_prefix
    )

