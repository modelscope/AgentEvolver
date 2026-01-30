import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from games.web.game_state_manager import GameStateManager


def test_stop_game_sets_flags_and_status():
    manager = GameStateManager()
    manager.set_mode("observe", None, game="avalon")
    manager.update_game_state(status="running")

    manager.stop_game()

    assert manager.should_stop is True
    assert manager.game_state["status"] == "stopped"


def test_reset_clears_stop_flag_and_sets_waiting():
    manager = GameStateManager()
    manager.set_mode("participate", "0", game="avalon")
    manager.update_game_state(status="running", phase=1, mission_id=2, round_id=3)
    manager.stop_game()

    manager.reset()

    assert manager.should_stop is False
    assert manager.game_state["status"] == "waiting"
    assert manager.game_state["game"] == "avalon"
    assert manager.game_state["phase"] is None
    assert manager.game_state["mission_id"] is None
    assert manager.game_state["round_id"] is None
