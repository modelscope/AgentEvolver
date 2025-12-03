# -*- coding: utf-8 -*-
"""Test script for AvalonTrainer."""
import argparse
import asyncio
import os
import sys
import json
from pathlib import Path
from typing import Optional

# Add BeyondAgent directory to path for imports
astune_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(astune_dir))

from games.avalon.trainer import AvalonTrainer


async def test_trainer(
    task_config_path: str,
    model_path: str,
    output_dir: Optional[str] = None,
):
    """Test AvalonTrainer by running a training game and collecting data.
    
    Args:
        task_config_path: Path to task config YAML file.
        model_path: Path to the model to be trained.
        output_dir: Optional directory to save training results. If None, prints to console.
    """
    print("=" * 80)
    print("Testing AvalonTrainer")
    print("=" * 80)
    print(f"Task config: {task_config_path}")
    print(f"Model path: {model_path}")
    print("=" * 80)
    
    try:
        # Create trainer
        trainer = AvalonTrainer(
            task_config=task_config_path,
            model_path=model_path,
        )
        
        print(f"\nTrainer initialized:")
        print(f"  - Number of players: {trainer.num_players}")
        print(f"  - Language: {trainer.language}")
        print(f"  - Log directory: {trainer.log_dir}")
        print(f"  - Train roles: {trainer.train_roles}")
        print(f"  - Default model: {trainer.default_model_name}")
        print(f"  - Training model: {model_path}")
        
        # Run training
        print("\n" + "=" * 80)
        print("Starting training game...")
        print("=" * 80)
        
        result = await trainer.train()
        
        # Print results
        print("\n" + "=" * 80)
        print("Training Results")
        print("=" * 80)
        print(f"Reward: {result['reward']}")
        print(f"Number of model calls: {len(result['model_call_history'])}")
        
        # Print model call history summary
        if result['model_call_history']:
            print("\nModel Call History Summary:")
            for i, call in enumerate(result['model_call_history'][:5]):  # Show first 5
                print(f"\n  Call {i+1}:")
                prompt_preview = call.get('prompt', '')[:100] if call.get('prompt') else 'N/A'
                response_preview = call.get('response', '')[:100] if call.get('response') else 'N/A'
                print(f"    Prompt: {prompt_preview}...")
                print(f"    Response: {response_preview}...")
            
            if len(result['model_call_history']) > 5:
                print(f"\n  ... and {len(result['model_call_history']) - 5} more calls")
        else:
            print("\nNo model call history recorded.")
        
        # Save results if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, "training_result.json")
            
            # Prepare output data
            output_data = {
                'reward': result['reward'],
                'num_model_calls': len(result['model_call_history']),
                'model_call_history': result['model_call_history'],
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            print(f"\nResults saved to: {output_file}")
        
        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print("=" * 80)
        
        return result
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("Error occurred during training:")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test AvalonTrainer"
    )
    parser.add_argument(
        "--task-config",
        type=str,
        default="games/avalon/task_config.yaml",
        help="Path to task config YAML file (default: games/avalon/task_config.yaml)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model to be trained (required)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save training results (optional)",
    )
    
    args = parser.parse_args()
    
    # Resolve task config path
    # Priority: 1. Absolute path, 2. Relative to BeyondAgent root, 3. Relative to current working directory
    task_config_path = args.task_config
    if not os.path.isabs(task_config_path):
        # Get BeyondAgent root directory (parent of games directory)
        beyond_agent_root = Path(__file__).parent.parent.parent
        # Try relative to BeyondAgent root first
        task_config_path_abs = beyond_agent_root / task_config_path
        if task_config_path_abs.exists():
            task_config_path = str(task_config_path_abs)
        else:
            # Try relative to current working directory
            cwd_path = Path.cwd() / task_config_path
            if cwd_path.exists():
                task_config_path = str(cwd_path)
            else:
                # Try relative to current file location
                file_path = Path(__file__).parent / task_config_path
                if file_path.exists():
                    task_config_path = str(file_path)
                # Otherwise, keep original path and let trainer handle it
    
    print(f"Resolved task config path: {task_config_path}")
    if not os.path.exists(task_config_path):
        print(f"Warning: Task config file not found at: {task_config_path}")
    
    asyncio.run(
        test_trainer(
            task_config_path=task_config_path,
            model_path=args.model_path,
            output_dir=args.output_dir,
        )
    )
