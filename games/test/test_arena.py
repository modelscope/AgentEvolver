#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple test script for arena leaderboard system."""
import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from games.evaluation.leaderboard.leaderboard_db import LeaderboardDB
from games.evaluation.leaderboard.leaderboard import calculate_elo, generate_leaderboard_from_db


def test_leaderboard_db():
    """Test LeaderboardDB basic functionality."""
    print("=" * 60)
    print("Testing LeaderboardDB")
    print("=" * 60)
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        db_path = f.name
    
    try:
        # Initialize database
        db = LeaderboardDB(db_path)
        print(f"✓ Created database at: {db_path}")
        
        # Add models
        models = ['qwen-plus', 'qwen-max', 'qwen2.5-14b', 'qwen2.5-32b']
        for model in models:
            db.add_model(model)
        print(f"✓ Added {len(models)} models")
        
        # Simulate game results
        print("\nSimulating 10 games...")
        for game_id in range(10):
            # Simulate a game result (Avalon: 5 players, good wins)
            result = {
                'roles': [
                    {'role_name': 'Merlin_0', 'model_name': 'qwen-plus', 'score': 1},
                    {'role_name': 'Servant_0', 'model_name': 'qwen-max', 'score': 1},
                    {'role_name': 'Servant_1', 'model_name': 'qwen2.5-14b', 'score': 1},
                    {'role_name': 'Assassin_0', 'model_name': 'qwen2.5-32b', 'score': 0},
                    {'role_name': 'Minion_0', 'model_name': 'qwen-plus', 'score': 0},
                ]
            }
            db.update_from_game_results([result])
        
        print(f"✓ Updated database with 10 games")
        
        # Check statistics
        balance = db.get_game_count_balance()
        print(f"\nGame Count Balance:")
        print(f"  Min: {balance['min']}")
        print(f"  Max: {balance['max']}")
        print(f"  Mean: {balance['mean']}")
        print(f"  Balance Ratio: {balance['balance_ratio']:.3f}")
        
        # Get leaderboard data
        leaderboard_data = db.get_leaderboard_data()
        print(f"\nLeaderboard Summary:")
        print(f"  Total Games: {leaderboard_data['total_games']}")
        print(f"  Models: {len(leaderboard_data['models'])}")
        
        for model, stats in leaderboard_data['models'].items():
            print(f"  {model}: Elo={stats['elo']:.0f}, Games={stats['total_games']}, WinRate={stats['win_rate']:.1f}%")
        
        # Test Elo calculation
        print("\n" + "=" * 60)
        print("Testing Elo Calculation")
        print("=" * 60)
        elo_a, elo_b = 1500, 1500
        new_elo_a, new_elo_b = calculate_elo(elo_a, elo_b, 1.0, k=32)
        print(f"Initial: A={elo_a}, B={elo_b}")
        print(f"After A wins: A={new_elo_a:.1f}, B={new_elo_b:.1f}")
        
        elo_a, elo_b = new_elo_a, new_elo_b
        new_elo_a, new_elo_b = calculate_elo(elo_a, elo_b, 0.0, k=32)
        print(f"After B wins: A={new_elo_a:.1f}, B={new_elo_b:.1f}")
        
        # Test leaderboard display
        print("\n" + "=" * 60)
        print("Testing Leaderboard Display")
        print("=" * 60)
        arena_config = {'models': models}
        leaderboard_str = generate_leaderboard_from_db(leaderboard_data, arena_config, 'avalon')
        print(leaderboard_str)
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
    finally:
        # Cleanup
        Path(db_path).unlink(missing_ok=True)
        print(f"\nCleaned up test database: {db_path}")


if __name__ == "__main__":
    test_leaderboard_db()

