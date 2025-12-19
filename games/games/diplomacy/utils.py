# -*- coding: utf-8 -*-
import os
import json
import numpy as np
from typing import Dict, Any, List
from xml.dom import minidom
from diplomacy import Game

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def add_legend_to_svg(svg_content: str, colors: Dict[str, str]) -> str:
    """
    Add a power color legend to the SVG content.
    向 SVG 内容添加势力颜色图例。
    """
    try:
        doc = minidom.parseString(svg_content)
        svg = doc.getElementsByTagName('svg')[0]
        
        # Get viewBox dimensions first as they define the coordinate system
        viewBox = svg.getAttribute('viewBox')
        vb_min_x = 0
        vb_min_y = 0
        vb_width = 0
        vb_height = 0
        
        if viewBox:
            vb_parts = [float(x) for x in viewBox.split()]
            if len(vb_parts) == 4:
                vb_min_x = vb_parts[0]
                vb_min_y = vb_parts[1]
                vb_width = vb_parts[2]
                vb_height = vb_parts[3]
        
        # Fallback to width/height attributes if viewBox missing
        if not vb_width:
            def parse_dim(val):
                if not val: return 1000.0
                return float(''.join(filter(lambda x: x.isdigit() or x == '.', val)))
            vb_width = parse_dim(svg.getAttribute('width'))
            vb_height = parse_dim(svg.getAttribute('height'))

        # Legend settings (Horizontal Bottom)
        legend_height = 70  # Height to add at bottom
        box_size = 28       # Larger box
        font_size = 28      # Larger font
        
        # Calculate new total height
        new_vb_height = vb_height + legend_height
        
        # Update viewBox
        svg.setAttribute('viewBox', f"{vb_min_x} {vb_min_y} {vb_width} {new_vb_height}")
        
        # Also update height attribute if it exists
        current_height_attr = svg.getAttribute('height')
        if current_height_attr and current_height_attr.replace('.','').isdigit():
             svg.setAttribute('height', str(float(current_height_attr) + legend_height))

        # Create Legend Group
        legend_g = doc.createElement('g')
        legend_g.setAttribute('id', 'LegendLayer')
        
        # Position: Bottom of the ORIGINAL map area
        legend_g.setAttribute('transform', f'translate({vb_min_x}, {vb_min_y + vb_height})')
        
        # Calculate horizontal spacing
        num_powers = len(colors)
        if num_powers > 0:
            item_width = vb_width / num_powers
        else:
            item_width = 100

        sorted_colors = sorted(colors.items())
        
        for i, (power, color) in enumerate(sorted_colors):
            # Start x for this item
            item_start_x = i * item_width
            
            # Box
            rect = doc.createElement('rect')
            rect.setAttribute('x', str(item_start_x + 10))
            rect.setAttribute('y', str((legend_height - box_size) / 2))
            rect.setAttribute('width', str(box_size))
            rect.setAttribute('height', str(box_size))
            rect.setAttribute('fill', color)
            rect.setAttribute('stroke', '#666')
            rect.setAttribute('stroke-width', '0.5')
            legend_g.appendChild(rect)
            
            # Text
            text = doc.createElement('text')
            text.setAttribute('x', str(item_start_x + 10 + box_size + 8))
            text.setAttribute('y', str((legend_height + font_size) / 2 - 2))
            text.setAttribute('font-family', 'Arial, sans-serif')
            text.setAttribute('font-size', str(font_size))
            text.setAttribute('fill', '#FFFFFF')
            text.setAttribute('font-weight', 'bold')
            text.appendChild(doc.createTextNode(power))
            legend_g.appendChild(text)
            
        svg.appendChild(legend_g)
        return doc.toxml()
    except Exception as e:
        print(f"{Colors.WARNING}Failed to add legend to SVG: {e}{Colors.ENDC}")
        return svg_content

async def save_game_logs(
    agents: List[Any],
    game: Game,
    game_log: Dict[str, Any],
    game_log_dir: str,
) -> None:
    """
    保存游戏日志，包括 Agent 的记忆和游戏进程日志。
    """
    def convert_to_serializable(obj: Any) -> Any:
        """Convert numpy types to Python native types."""
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    # Extract model names from agents if available
    model_names = []
    if agents is not None:
        for agent in agents:
            model_name = "Unknown"
            try:
                # Try to get model name from agent.model
                if hasattr(agent, 'model') and agent.model is not None:
                    if hasattr(agent.model, 'model_name'):
                        model_name = agent.model.model_name
                    elif hasattr(agent.model, 'name'):
                        model_name = agent.model.name
            except Exception:
                pass
            model_names.append(model_name)
    
    game_log_data = {
        "map_name": game.map_name,
        "game_id": game.game_id,
        "outcome": game.outcome,
        "game_log": convert_to_serializable(game_log),
    }
    
    # Add model names if available
    if model_names:
        game_log_data["model_names"] = model_names
    
    os.makedirs(game_log_dir, exist_ok=True)
    game_log_path = os.path.join(game_log_dir, "game_log.json")
    with open(game_log_path, 'w', encoding='utf-8') as f:
        json.dump(game_log_data, f, ensure_ascii=False, indent=2)
    # print(f"{Colors.OKBLUE}Game log saved to {game_log_path}{Colors.ENDC}")
    
    # Save each agent's memory
    for agent in agents:
        try:
            if hasattr(agent, 'memory') and agent.memory is not None:
                agent_memory = await agent.memory.get_memory()
                memory_data = {
                    "agent_name": agent.name,
                    "power_name": getattr(agent, 'power_name', 'Unknown'),
                    "memory_count": len(agent_memory),
                    "memory": [msg.to_dict() for msg in agent_memory],
                }
                
                memory_path = os.path.join(game_log_dir, f"{agent.name}_memory.json")
                with open(memory_path, 'w', encoding='utf-8') as f:
                    json.dump(memory_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"{Colors.WARNING}Failed to save memory for agent {agent.name}: {e}{Colors.ENDC}")

# 提取消息解析逻辑为独立方法
def parse_negotiation_messages(raw: str, power_name: str, power_names: List[str]) -> List[Dict]:
    """Parse negotiation messages from agent response."""
    raw = (raw or "").strip()
    parsed_msgs = []
    
    # (A) Try parse JSON list first
    json_text = raw
    json_text = re.sub(r"^\s*```(?:json)?\s*", "", json_text, flags=re.IGNORECASE)
    json_text = re.sub(r"\s*```\s*$", "", json_text).strip()
    
    start = json_text.find("[")
    json_list_sub = None
    if start >= 0:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(json_text)):
            ch = json_text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            else:
                if ch == '"':
                    in_str = True
                    continue
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        json_list_sub = json_text[start:i + 1]
                        break
    
    if json_list_sub:
        cand = (
            json_list_sub.replace(""", '"')
            .replace(""", '"')
            .replace("'", "'")
            .strip()
        )
        try:
            messages_data = json.loads(cand)
            if isinstance(messages_data, list):
                for msg_data in messages_data:
                    if not isinstance(msg_data, dict):
                        continue
                    content = (msg_data.get("content") or "").strip()
                    if not content:
                        continue
                    
                    mt = (msg_data.get("message_type") or "").lower().strip()
                    if mt not in ("private", "global"):
                        mt = "global" if ("recipient" not in msg_data) else "private"
                    
                    if mt == "global":
                        parsed_msgs.append({"message_type": "global", "recipient": "GLOBAL", "content": content})
                    else:
                        rec = (msg_data.get("recipient") or "").strip()
                        parsed_msgs.append({"message_type": "private", "recipient": rec, "content": content})
        except json.JSONDecodeError:
            pass
    
    # (B) If JSON failed: try "FRANCE: Hi" lines
    if not parsed_msgs:
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        for ln in lines:
            m = re.match(r"^([A-Za-z_]+)\s*:\s*(.+)$", ln)
            if not m:
                continue
            target = m.group(1).strip().upper()
            content = m.group(2).strip()
            if not content:
                continue
            if target in power_names or target in [p.upper() for p in power_names]:
                parsed_msgs.append({"message_type": "private", "recipient": target, "content": content})
    
    # (C) Fallback: plain text -> GLOBAL
    if not parsed_msgs and raw:
        parsed_msgs = [{"message_type": "global", "recipient": "GLOBAL", "content": raw}]
    
    return parsed_msgs

def order_to_natural_language(order: str) -> str:
    """
    将标准的 Diplomacy 指令字符串转换为中文自然语言描述。
    """
    if ' S ' in order:
        unit, target = order.split(' S ', 1)
        return f"{unit} 支援 {target}"
    elif ' C ' in order:
        unit, target = order.split(' C ', 1)
        return f"{unit} 护送 {target}"
    elif ' - ' in order:
        unit, dest = order.split(' - ', 1)
        return f"{unit} 移动到 {dest}"
    elif order.endswith(' H'):
        return f"{order[:-2]} 原地驻守"
    elif ' R ' in order:
        unit, dest = order.split(' R ', 1)
        return f"{unit} 撤退到 {dest}"
    elif ' D' in order:
        return f"{order.replace(' D', '')} 解散"
    elif ' B' in order:
        return f"{order.replace(' B', '')} 募集"
    return order


# -*- coding: utf-8 -*-
"""Utility functions for the Avalon game."""
import re
from typing import Any

from agentscope.agent import AgentBase, ReActAgent
from agentscope.message import Msg

def load_prompts(language: str = "en") -> dict:
    """
    加载prompt文本，支持中英文。
    """
    base_path = os.path.join(os.path.dirname(__file__), "prompt")
    lang = (language or "").lower().strip()
    # Web 端使用 zh；历史配置里也可能是 zn/cn/chinese 等
    zh_aliases = {"zh", "zn", "cn", "zh-cn", "zh_cn", "chinese"}
    if lang in zh_aliases or (os.getenv("LANGUAGE", "").lower().strip() in zh_aliases):
        prompts_dir = os.path.join(base_path, "prompts_simple_zn")
    else:
        prompts_dir = os.path.join(base_path, "prompts_simple")
    prompts = {}
    if os.path.isdir(prompts_dir):
        for fname in os.listdir(prompts_dir):
            full = os.path.join(prompts_dir, fname)
            if os.path.isfile(full) and fname.endswith('.txt'):
                try:
                    with open(full, 'r', encoding='utf-8') as f:
                        prompts[fname] = f.read()
                except Exception:
                    prompts[fname] = ''
    else:
        print(f"[WARNING] 未找到 prompts 目录: {prompts_dir}")
    return prompts

# -*- coding: utf-8 -*-
"""Utility functions and classes for the Avalon game."""
import json
import os
import re
from datetime import datetime
from typing import Any

import numpy as np
from agentscope.agent import AgentBase
from loguru import logger


# ============================================================================
# Parser
# ============================================================================

class Parser:
    """Parser class for parsing agent responses."""
    
    APPROVE_KEYWORDS = ['yes', 'approve', 'accept', '是', '同意', '通过']
    REJECT_KEYWORDS = ['no', 'reject', '否', '拒绝', '不同意']
        
    @staticmethod
    def extract_text_from_content(content: str | list) -> str:
        """Extract text string from agentscope message content."""
        if isinstance(content, str):
            return content
        
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content", "")))
                else:
                    parts.append(str(item))
            return " ".join(parts)
        
        return str(content)
    
    @staticmethod
    def parse_team_from_response(response: str | list) -> list[int]:
        """Parse team list from agent response."""
        text = Parser.extract_text_from_content(response)
        
        # Try to find list pattern like [0, 1, 2]
        list_match = re.search(r'$$[\s]*\d+[\s]*(?:,[\s]*\d+[\s]*)*$$', text)
        if list_match:
            return [int(n) for n in re.findall(r'\d+', list_match.group())]
        
        # Fallback: extract all numbers (limit to 10 players)
        return [int(n) for n in re.findall(r'\d+', text)[:10]]
    
    @staticmethod
    def parse_vote_from_response(response: str | list) -> int:
        """Parse vote (0 or 1) from agent response."""
        text = Parser.extract_text_from_content(response).lower().strip()
        
        if any(kw in text for kw in Parser.APPROVE_KEYWORDS):
            return 1
        return 0  # Default to reject
    
    @staticmethod
    def parse_player_id_from_response(response: str | list, max_id: int) -> int:
        """Parse player ID from agent response."""
        text = Parser.extract_text_from_content(response)
        numbers = re.findall(r'\d+', text)
        
        if numbers:
            return max(0, min(int(numbers[-1]), max_id))
        return 0


# ============================================================================
# Logger
# ============================================================================

class GameLogger:
    """Logger class for game logging functionality."""
    
    def __init__(self):
        self.game_log = {
            "initialization": {},
            "missions": [],
            "assassination": None,
            "game_end": None,
        }
        self.game_log_dir = None
    
    def initialize_game_log(self, roles: list[tuple], num_players: int) -> None:
        """Initialize game log with roles and player count."""
        self.game_log["initialization"] = {
            "roles": [(role_id, role_name, side) for role_id, role_name, side in roles],
            "num_players": num_players,
        }
    
    def create_game_log_dir(self, log_dir: str | None) -> str | None:
        """Create game log directory and return the path."""
        if not log_dir:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.game_log_dir = os.path.join(log_dir, f"game_{timestamp}")
        os.makedirs(self.game_log_dir, exist_ok=True)
        logger.info(f"Game logs will be saved to: {self.game_log_dir}")
        return self.game_log_dir
    
    def add_mission(self, mission_id: int, round_id: int, leader: int) -> None:
        """Add a new mission entry to the game log."""
        if not self.game_log_dir:
            return
        
        self.game_log["missions"].append({
            "mission_id": mission_id,
            "round_id": round_id,
            "leader": leader,
            "discussion": [],
            "team_proposed": [],
        })
    
    def add_discussion_messages(self, discussion_msgs: list[dict]) -> None:
        """Add discussion messages to the current mission."""
        if self.game_log_dir and self.game_log["missions"]:
            self.game_log["missions"][-1]["discussion"] = discussion_msgs
    
    def add_team_proposal(self, team: list[int]) -> None:
        """Add team proposal to the current mission."""
        if self.game_log_dir and self.game_log["missions"]:
            self.game_log["missions"][-1]["team_proposed"] = team
    
    def add_team_voting(self, team: list[int], votes: list[int], approved: bool) -> None:
        """Add team voting results to the current mission."""
        if self.game_log_dir and self.game_log["missions"]:
            self.game_log["missions"][-1]["team_voting"] = {
                "team": team,
                "votes": votes,
                "approved": approved,
            }
    
    def add_quest_voting(self, team: list[int], votes: list[int], num_fails: int, succeeded: bool) -> None:
        """Add quest voting results to the current mission."""
        if self.game_log_dir and self.game_log["missions"]:
            self.game_log["missions"][-1]["quest_voting"] = {
                "team": team,
                "votes": votes,
                "num_fails": num_fails,
                "succeeded": succeeded,
            }
    
    def add_assassination(self, assassin_id: int, target: int, good_wins: bool) -> None:
        """Add assassination results to the game log."""
        if self.game_log_dir:
            self.game_log["assassination"] = {
                "assassin_id": assassin_id,
                "target": target,
                "good_wins": good_wins,
            }
    
    @staticmethod
    def _convert_to_serializable(obj: Any) -> Any:
        """Convert numpy types to Python native types."""
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: GameLogger._convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [GameLogger._convert_to_serializable(item) for item in obj]
        return obj
    
    async def save_game_logs(self, agents: list[AgentBase], env: Any, roles: list[tuple]) -> None:
        """Save game logs including agent memories and game log."""
        if not self.game_log_dir:
            return
        
        self._save_game_log_json(env, roles)
        await self._save_agent_memories(agents, roles)
    
    def _save_game_log_json(self, env: Any, roles: list[tuple]) -> None:
        """Save game log to JSON file."""
        self.game_log["game_end"] = {
            "good_victory": env.good_victory,
            "quest_results": env.quest_results,
        }
        
        game_log_data = {
            "roles": [(int(r), n, bool(s)) for r, n, s in roles],
            "game_result": {
                "good_victory": bool(env.good_victory),
                "quest_results": [bool(r) for r in env.quest_results],
            },
            "game_log": self._convert_to_serializable(self.game_log),
        }
        
        path = os.path.join(self.game_log_dir, "game_log.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(game_log_data, f, ensure_ascii=False, indent=2)
        # logger.info(f"Game log saved to {path}")
    
    async def _save_agent_memories(self, agents: list[AgentBase], roles: list[tuple]) -> None:
        """Save each agent's memory and model call history to separate JSON files."""
        for i, agent in enumerate(agents):
            try:
                agent_data = {
                    "agent_name": agent.name,
                    "agent_index": i,
                    "role": roles[i][1] if i < len(roles) else "Unknown",
                }
                
                # Save memory if available
                if hasattr(agent, 'memory') and agent.memory is not None:
                    agent_memory = await agent.memory.get_memory()
                    agent_data["memory_count"] = len(agent_memory)
                    agent_data["memory"] = [msg.to_dict() for msg in agent_memory]
                
                # Save model call history if available (for ThinkingReActAgent)
                if hasattr(agent, 'model_call_history'):
                    # Convert model call history to serializable format
                    serializable_history = []
                    for call_record in agent.model_call_history:
                        serializable_record = {
                            "prompt": call_record.get("prompt", ""),
                            "response": call_record.get("response", ""),
                            "response_msg": self._convert_to_serializable(call_record.get("response_msg", {})),
                        }
                        serializable_history.append(serializable_record)
                    agent_data["model_call_history"] = serializable_history
                    agent_data["model_call_count"] = len(serializable_history)
                    
                    # Log model call history to logger
                    logger.info(f"Agent {agent.name} model call history: {agent_data['model_call_count']} calls")
                    for idx, call_record in enumerate(serializable_history):
                        logger.debug(f"Agent {agent.name} call {idx + 1}: prompt length={len(call_record.get('prompt', ''))}, response length={len(call_record.get('response', ''))}")
                
                # Only save if we have data
                if "memory" in agent_data or "model_call_history" in agent_data:
                    path = os.path.join(self.game_log_dir, f"{agent.name}_memory.json")
                    with open(path, 'w', encoding='utf-8') as f:
                        json.dump(agent_data, f, ensure_ascii=False, indent=2)
                    logger.info(f"Agent {agent.name} memory and model calls saved to {path}")
            except Exception as e:
                logger.warning(f"Failed to save memory for agent {agent.name}: {e}")


# ============================================================================
# Language Formatter
# ============================================================================

class LanguageFormatter:
    """Language formatter helper to handle language-specific formatting."""
    
    LANGUAGE_CONFIG = {
        "zh": {
            "role_names": {
                "Merlin": "梅林", "Servant": "忠臣", "Assassin": "刺客", "Minion": "爪牙",
                "Percival": "派西维尔", "Morgana": "莫甘娜", "Mordred": "莫德雷德", "Oberon": "奥伯伦",
            },
            "side_names": {"Good": "好人", "Evil": "坏人"},
            "player_prefix": "玩家",
            "separator": "和",
            "vote_approve": "批准",
            "vote_reject": "拒绝",
            "is_text": "是",
        },
        "en": {
            "role_names": {},
            "side_names": {"Good": "Good", "Evil": "Evil"},
            "player_prefix": "Player",
            "separator": "and",
            "vote_approve": "Approve",
            "vote_reject": "Reject",
            "is_text": "is",
        }
    }
    
    def __init__(self, language: str = "en"):
        """Initialize language formatter with language code."""
        self.is_zh = language.lower() in ["zh", "cn", "chinese"]
        config = self.LANGUAGE_CONFIG["zh" if self.is_zh else "en"]
        
        self.role_names = config["role_names"]
        self.side_names = config["side_names"]
        self.player_prefix = config["player_prefix"]
        self.separator = config["separator"]
        self.vote_approve = config["vote_approve"]
        self.vote_reject = config["vote_reject"]
        self.is_text = config["is_text"]
    
    def format_player_name(self, agent_name: str) -> str:
        """Format player name (Player0 -> 玩家 0)."""
        if not self.is_zh or not agent_name.startswith("Player"):
            return agent_name
        
        return f"{self.player_prefix} {agent_name.replace('Player', '')}"
    
    def format_player_id(self, player_id: int) -> str:
        """Format player ID (0 -> '玩家 0' or 'Player 0')."""
        return f"{self.player_prefix} {player_id}"
    
    def format_role_name(self, role_name: str) -> str:
        """Format role name (Merlin -> 梅林)."""
        return self.role_names.get(role_name, role_name)
    
    def format_side_name(self, side: bool) -> str:
        """Format side name (True -> '好人' or 'Good')."""
        return self.side_names["Good" if side else "Evil"]
    
    def format_agents_names(self, agents: list[AgentBase]) -> str:
        """Format list of agent names for display."""
        if not agents:
            return ""
        
        names = [self.format_player_name(a.name) for a in agents]
        if len(names) == 1:
            return names[0]
        
        return f"{', '.join(names[:-1])} {self.separator} {names[-1]}"
    
    def format_vote_details(self, votes: list[int], approved: bool) -> tuple[str, str, str]:
        """Format vote details for display. Returns (votes_detail, result_text, outcome_text)."""
        approve_text = self.vote_approve
        reject_text = self.vote_reject
        result_text = approve_text if approved else reject_text
        
        votes_detail = ", ".join([
            f"{self.format_player_id(i)}: {result_text if v == approved else (reject_text if approved else approve_text)}"
            for i, v in enumerate(votes)
        ])
        
        outcome_text = result_text if self.is_zh else result_text.lower() + "d"
        
        return votes_detail, result_text, outcome_text
    
    def format_sides_info(self, roles: list[tuple]) -> list[str]:
        """Format sides information for visibility."""
        return [
            f"{self.format_player_id(j)} {self.is_text} {self.format_side_name(s)}"
            for j, (_, _, s) in enumerate(roles)
        ]
    
    def calculate_role_counts(self, config: Any) -> dict[str, Any]:
        """Calculate role counts for system prompt."""
        merlin_count = 1 if config.merlin else 0
        percival_count = 1 if config.percival else 0
        
        return {
            "num_players": config.num_players,
            "max_player_id": config.num_players - 1,
            "num_good": config.num_good,
            "merlin_count": merlin_count,
            "servant_count": config.num_good - merlin_count - percival_count,
            "num_evil": config.num_evil,
            "assassin_count": 1,
            "minion_count": config.num_evil - 1,
            "percival_count": percival_count,
        }
    
    def format_system_prompt(self, config: Any, prompts_class: Any) -> str:
        """Format system prompt with role counts."""
        return prompts_class.system_prompt_template.format(**self.calculate_role_counts(config))
    
    def format_true_roles(self, roles: list[tuple]) -> str:
        """Format true roles for game end display."""
        return ", ".join([
            f"{self.format_player_id(i)}: {self.format_role_name(role_name)}"
            for i, (_, role_name, _) in enumerate(roles)
        ])
    
    def format_game_end_message(self, good_victory: bool, roles: list[tuple], prompts_class: Any) -> str:
        """Format game end message with result and true roles."""
        result = prompts_class.to_all_good_wins if good_victory else prompts_class.to_all_evil_wins
        true_roles_str = self.format_true_roles(roles)
        return prompts_class.to_all_game_end.format(result=result, true_roles=true_roles_str)

def names_to_str(agents: list[str] | list[ReActAgent]) -> str:
    """Return a string of agent names."""
    if not agents:
        return ""
    
    names = [agent.name if isinstance(agent, ReActAgent) else agent for agent in agents]
    
    if len(names) == 1:
        return names[0]
    
    return ", ".join([*names[:-1], "and " + names[-1]])


class EchoAgent(AgentBase):
    """Echo agent that repeats the input message (Moderator for public announcements)."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "Moderator"

    async def reply(self, content: str) -> Msg:
        """Repeat the input content with its name and role (public moderator announcement)."""
        msg = Msg(
            self.name,
            content,
            role="assistant",
        )
        # Print with clear label for public information
        print(f"\n[MODERATOR PUBLIC INFO] {self.name}")
        print("-" * 70)
        await self.print(msg)
        print("-" * 70 + "\n")
        return msg

    async def handle_interrupt(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Msg:
        """Handle interrupt."""

    async def observe(self, msg: Msg | list[Msg] | None) -> None:
        """Observe the user's message."""


def parse_team_from_response(response: str | list) -> list[int]:
    """Parse team list from agent response."""
    response = extract_text_from_content(response)
    
    # Try to find list pattern like [0, 1, 2] or [0,1,2]
    list_match = re.search(r'\[[\s]*\d+[\s]*(?:,[\s]*\d+[\s]*)*\]', response)
    if list_match:
        return [int(n) for n in re.findall(r'\d+', list_match.group())]
    
    # Fallback: extract all numbers (limit to 10 players)
    numbers = re.findall(r'\d+', response)
    return [int(n) for n in numbers[:10]]


def parse_vote_from_response(response: str | list) -> int:
    """Parse vote (0 or 1) from agent response.
    
    Supports both English and Chinese responses:
    - Approve: yes, approve, accept, 1 (English) | 是, 批准, 同意, 通过, 赞成, 支持 (Chinese)
    - Reject: no, reject, 0 (English) | 否, 拒绝, 不同意, 失败, 反对 (Chinese)
    """
    response = extract_text_from_content(response)
    text_lower = response.lower().strip()
    text_original = response.strip()
    
    # Approve keywords: English (lowercase) + Chinese (original)
    approve_keywords = ['yes', 'approve', 'accept', '1', '是', '批准', '同意', '通过', '赞成', '支持', '一']
    # Reject keywords: English (lowercase) + Chinese (original)
    reject_keywords = ['no', 'reject', '0', '否', '拒绝', '不同意', '失败', '反对', '零']
    
    # Check both lowercase and original text
    for text in [text_lower, text_original]:
        if any(kw in text for kw in approve_keywords):
            return 1
        if any(kw in text for kw in reject_keywords):
            return 0
    
    return 0  # Default to reject


def parse_player_id_from_response(response: str | list, max_id: int) -> int:
    """Parse player ID from agent response."""
    # Extract text from content (handles both str and list formats)
    response = extract_text_from_content(response)
    
    numbers = re.findall(r'\d+', response)
    if numbers:
        player_id = int(numbers[-1])  # Take the last number
        return max(0, min(player_id, max_id))
    return 0


def extract_text_from_content(content: str | list) -> str:
    """Extract text string from agentscope message content (handles both str and list formats)."""
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                text_parts.append(str(item.get("text") or item.get("content", "")))
            elif isinstance(item, str):
                text_parts.append(item)
        return " ".join(text_parts)
    return str(content)


def remove_redacted_reasoning(text: str | list) -> str:
    """
    Remove <think>...</think> or <think>...</think> tags from text.
    
    Args:
        text: Text that may contain redacted reasoning tags (can be str or list).
        
    Returns:
        Text with redacted reasoning removed.
    """
    # Extract text if content is a list
    text = extract_text_from_content(text)
    
    # Support both <think> and <think> tags
    REDACTED_PATTERN = r'<(?:redacted_reasoning|think)>.*?</(?:redacted_reasoning|think)>'
    MULTI_NEWLINE_PATTERN = r'\n\s*\n\s*\n'
    
    result = re.sub(REDACTED_PATTERN, '', text, flags=re.DOTALL | re.IGNORECASE)
    result = re.sub(MULTI_NEWLINE_PATTERN, '\n\n', result)
    return result.strip()

