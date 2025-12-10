# -*- coding: utf-8 -*-
import re
import random
import os
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from diplomacy import Game
from diplomacy.engine.renderer import Renderer
from agentscope.message import Msg
from agentscope.agent import AgentBase

# 引入重构后的工具函数
from .utils import Colors, add_legend_to_svg, save_game_logs, order_to_natural_language, load_prompts

@dataclass
class DiplomacyConfig:
    """Configuration for Diplomacy game."""
    max_phases: int = 10
    map_name: str = 'standard'
    seed: int = 42
    num_negotiation_rounds: int = 0
    debug: bool = False

class DiplomacyGame:
    """
    Diplomacy game class that integrates all game functionality.
    Refactored to match AvalonGame structure.
    """
    
    def __init__(
        self,
        agents: List[AgentBase],
        config: DiplomacyConfig,
        log_dir: str | None = None,
        language: str = "en",
        observe_agent: AgentBase | None = None,
        state_manager: Any = None,
    ):
        """
        Initialize Diplomacy game.
        
        Args:
            agents: List of agents.
            config: Game configuration.
            log_dir: Directory to save game logs.
            language: Language for prompts (default "en").
            observe_agent: Optional observer agent.
            state_manager: Optional state manager for web mode.
        """
        self.agents = agents
        self.config = config
        self.log_dir = log_dir
        self.language = language
        self.observe_agent = observe_agent
        self.state_manager = state_manager
        
        # Initialize Game
        print(f"{Colors.HEADER}=== 初始化 Diplomacy (Map: {self.config.map_name}, Seed: {self.config.seed}) ==={Colors.ENDC}")
        self.game = Game(map_name=self.config.map_name, seed=self.config.seed)
        
        # Initialize Logging
        self.game_log = {
            "initialization": {
                "map_name": self.config.map_name,
                "seed": self.config.seed,
                "powers": list(self.game.powers.keys()),
            },
            "phases": [],
        }
        self.game_log_dir = None
        if self.log_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.game_log_dir = os.path.join(self.log_dir, f"diplomacy_game_{timestamp}")
            os.makedirs(self.game_log_dir, exist_ok=True)
            print(f"{Colors.OKBLUE}Game logs will be saved to: {self.game_log_dir}{Colors.ENDC}")

        # Load Prompts
        self.prompts = load_prompts(self.language)
        
        # Map powers to agents
        self.power_agent_map = {}
        self._assign_agents()

    def _assign_agents(self):
        """Assign powers to agents."""
        power_names = list(self.game.powers.keys())
        for i, power_name in enumerate(power_names):
            if i < len(self.agents):
                self.agents[i].power_name = power_name 
                self.power_agent_map[power_name] = self.agents[i]
            else:
                print(f"  {power_name} 没有分配 Agent (将随机行动)")

    async def _broadcast(self, content: str, sender: str = "System"):
        """Broadcast message to observe_agent and state_manager."""
        if self.observe_agent:
            msg = Msg(name=sender, content=content, role="assistant")
            await self.observe_agent.observe(msg)
        
    def _get_context_str(self, power_name: str) -> str:
        """Generate context string for a specific power."""
        power = self.game.powers[power_name]
        
        all_unit_locations = []
        for p_name, p_obj in self.game.powers.items():
            all_unit_locations.append(f"{p_name}: {p_obj.units}")
        all_unit_locations_str = "\n".join(all_unit_locations)

        all_supply_centers = []
        for p_name, p_obj in self.game.powers.items():
            all_supply_centers.append(f"{p_name}: {p_obj.centers}")
        all_supply_centers_str = "\n".join(all_supply_centers)

        possible_orders = self.game.get_all_possible_orders()
        orderable_locations = self.game.get_orderable_locations(power_name)
        possible_orders_str = ""
        if orderable_locations:
            for loc in orderable_locations:
                if loc in possible_orders and possible_orders[loc]:
                    possible_orders_str += f"  {loc}: {possible_orders[loc]}\n"
        else:
            possible_orders_str = "None"

        context_template = self.prompts.get('context_prompt.txt', '')
        return context_template.format(
            power_name=power_name,
            current_phase=self.game.get_current_phase(),
            home_centers=str(power.homes),
            agent_goals="Survive and expand.",
            order_history=self.game.order_history,
            all_unit_locations=all_unit_locations_str,
            all_supply_centers=all_supply_centers_str,
            possible_orders=possible_orders_str,
        )

    async def _initialize_agents(self):
        """Send initial system prompts to agents."""
        for power_name, agent in self.power_agent_map.items():
            base_sys = self.prompts.get('system_prompt.txt', '')
            country_sys = self.prompts.get(f"{power_name.lower()}_system_prompt.txt", '')
            few_shot = self.prompts.get('few_shot_example.txt', '')
            
            full_sys_prompt = f"{base_sys}---COUNTRY_SYSTEM---\n{country_sys}\n\n---FEW_SHOT---\n{few_shot}"
            
            sys_msg = Msg(name="Moderator", content=full_sys_prompt, role="assistant")
            await agent.observe(sys_msg)

            context_str = self._get_context_str(power_name)
            await agent.observe(Msg(name="Moderator", content=context_str, role="assistant"))

    async def _render_map(self, phase_name: str, save: bool = True):
        """Render the map and update state manager."""
        try:
            output_dir = os.path.join(os.path.dirname(__file__), 'images')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            renderer = Renderer(self.game)
            
            # Render with abbreviations
            svg_content = renderer.render(output_path=None, incl_abbrev=True)
            
            if svg_content:
                svg_content = add_legend_to_svg(svg_content, renderer.metadata['color'])
                
                if save:
                    filename = f"phase_{phase_name}.svg"
                    output_path = os.path.join(output_dir, filename)
                    with open(output_path, 'w') as f:
                        f.write(svg_content)
            
            if self.state_manager:
                sc_counts = {p: len(power.centers) for p, power in self.game.powers.items()}

                self.state_manager.update_game_state(
                    map_svg=svg_content,
                    sc_counts=sc_counts,
                )

            
            return svg_content
        except Exception as e:
            print(f"{Colors.WARNING}Map rendering failed: {e}{Colors.ENDC}")
            return None

    async def run(self) -> Game:
        """Run the Diplomacy game loop."""
        if self.state_manager:
            self.state_manager.update_game_state(status="running")

        await self._initialize_agents()
        await self._render_map(f"init_{self.game.get_current_phase()}")
        if self.state_manager:
            self.state_manager.save_history_snapshot(kind="init")
        print(f"{Colors.OKBLUE}[Renderer] 初始地图已渲染并发送至 Web 端。{Colors.ENDC}")

        phases_processed = 0
        
        while not self.game.is_game_done and phases_processed < self.config.max_phases:
            # Check stop flag
            if self.state_manager and getattr(self.state_manager, 'should_stop', False):
                print("Game stopped by user request")
                break

            current_phase = self.game.get_current_phase()
            print(f"\n{Colors.BOLD}{Colors.OKCYAN}--- 当前阶段: {current_phase} (第 {phases_processed + 1} 轮) ---{Colors.ENDC}")
            
            await self._broadcast(f"--- Phase: {current_phase} (Round {phases_processed + 1}) ---")
            
            if self.state_manager:
                self.state_manager.update_game_state(
                    phase=current_phase,
                    round=phases_processed + 1
                )
            
            phase_log = {
                "phase": current_phase,
                "round": phases_processed + 1,
                "negotiation": [],
                "orders": {},
                "order_logs": {},
                "sc_counts": {},
            }
            self.game_log["phases"].append(phase_log)

            # Negotiation Phase
            if current_phase.endswith('M') and self.config.num_negotiation_rounds > 0:
                await self._handle_negotiation_phase(current_phase, phases_processed, phase_log)

            # Order Phase
            await self._handle_order_phase(current_phase, phase_log)
            await self._render_map(f"orders_{current_phase}")
            if self.state_manager:
                self.state_manager.save_history_snapshot(kind="orders")
            # Process and Render
            self.game.process()
            phases_processed += 1
            
            sc_counts = {p: len(power.centers) for p, power in self.game.powers.items() if not power.is_eliminated()}
            phase_log["sc_counts"] = sc_counts
            print(f"{Colors.HEADER}各方补给中心: {sc_counts}{Colors.ENDC}")
            
            await self._render_map(f"result_{current_phase}")
            if self.state_manager:
                self.state_manager.save_history_snapshot(kind="result")

        print(f"\n{Colors.HEADER}=== 游戏结束 ==={Colors.ENDC}")
        print(f"{Colors.HEADER}结果: {self.game.outcome}{Colors.ENDC}")
        
        await self._broadcast(f"Game Over. Outcome: {self.game.outcome}")
        
        if self.state_manager:
            self.state_manager.update_game_state(status="finished")
        
        if self.game_log_dir:
            await save_game_logs(self.agents, self.game, self.game_log, self.game_log_dir)

        return self.game

    async def _handle_negotiation_phase(self, current_phase, round_num, phase_log):
        print(f"{Colors.OKBLUE}--- 开始谈判阶段 ({self.config.num_negotiation_rounds} 轮) ---{Colors.ENDC}")
        for round_idx in range(self.config.num_negotiation_rounds):
            print(f"{Colors.OKBLUE}  第{round_idx + 1}轮谈判{Colors.ENDC}")
            
            round_negotiation_log = {"round_idx": round_idx + 1, "messages": []}
            round_messages = [] 
            
            async def process_agent_negotiation(power_name, agent):
                agent_messages = []
                phase_instruction = self.prompts.get('conversation_instructions.txt', '')
                
                negotiation_prompt = (
                    "---PHASE_INSTRUCTIONS---\n"
                    f"Current Phase: {current_phase}\n"
                    f"Negotiation Round: {round_idx + 1}/{self.config.num_negotiation_rounds}\n"
                    f"{phase_instruction}"
                )
                
                msg = Msg(name="Moderator", content=negotiation_prompt, role="assistant")
                response_msg = await agent(msg)
                response_text = response_msg.get_text_content()

                # Parse messages
                # =========================
                # Robust message parsing (DROP-IN REPLACEMENT)
                # Supports:
                #  1) JSON list (possibly wrapped in ```json ... ``` or with extra text)
                #  2) "FRANCE: Hi" / multi-lines "FRANCE: Hi\nENGLAND: Hi"  -> private per line
                #  3) plain "hi" -> GLOBAL
                # =========================

                raw = (response_text or "").strip()
                parsed_msgs = []

                # ---- (A) Try parse JSON list first ----
                json_text = raw

                # Strip code fences if any
                json_text = re.sub(r"^\s*```(?:json)?\s*", "", json_text, flags=re.IGNORECASE)
                json_text = re.sub(r"\s*```\s*$", "", json_text).strip()

                # Extract the first complete JSON list by bracket counting (more robust than greedy regex)
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
                    # Normalize common “smart quotes”
                    cand = (
                        json_list_sub.replace("“", '"')
                        .replace("”", '"')
                        .replace("’", "'")
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
                                # tolerate missing/unknown
                                if mt not in ("private", "global"):
                                    mt = "global" if ("recipient" not in msg_data) else "private"

                                if mt == "global":
                                    parsed_msgs.append({"message_type": "global", "recipient": "GLOBAL", "content": content})
                                else:
                                    rec = (msg_data.get("recipient") or "").strip()
                                    parsed_msgs.append({"message_type": "private", "recipient": rec, "content": content})
                    except json.JSONDecodeError:
                        pass

                # ---- (B) If JSON failed / empty: try "FRANCE: Hi" lines ----
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
                        # Only accept real powers to avoid accidental matches
                        if target in self.game.powers.keys() or target in [p.upper() for p in self.game.powers.keys()]:
                            parsed_msgs.append({"message_type": "private", "recipient": target, "content": content})

                # ---- (C) Fallback: plain text -> GLOBAL ----
                if not parsed_msgs and raw:
                    parsed_msgs = [{"message_type": "global", "recipient": "GLOBAL", "content": raw}]

                # ---- Emit agent_messages (keep your original resolution logic) ----
                for msg_data in parsed_msgs:
                    content = msg_data["content"]
                    target = msg_data.get("recipient", "GLOBAL")
                    mt = msg_data.get("message_type", "global")

                    # Normalize message_type->target
                    if mt == "global":
                        target = "GLOBAL"

                    # Resolve target name (keep your original logic)
                    real_target = "GLOBAL"
                    if target != "GLOBAL":
                        for p in self.game.powers.keys():
                            if p.lower() == str(target).lower():
                                real_target = p
                                break

                    agent_messages.append((power_name, real_target, content))

                return agent_messages

            tasks = []
            for power_name, power in self.game.powers.items():
                if power.is_eliminated() or power_name not in self.power_agent_map:
                    continue
                tasks.append(process_agent_negotiation(power_name, self.power_agent_map[power_name]))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for res in results:
                for sender, recipient, content in res:
                    round_messages.append((sender, recipient, content))
                    round_negotiation_log["messages"].append({"sender": sender, "recipient": recipient, "content": content})
                    log_msg = f"{sender} -> {recipient}: {content}"
                    print(f"{Colors.OKGREEN}{log_msg}{Colors.ENDC}")
                    
                    # Broadcast to observer
                    await self._broadcast(f"{log_msg}", sender=sender) #给obs看

            phase_log["negotiation"].append(round_negotiation_log)

            # Deliver messages
            for sender, recipient, content in round_messages:
                if recipient in self.power_agent_map:
                    await self.power_agent_map[recipient].observe(Msg(name=sender, content=f"{sender} -> {recipient}: {content}", role="assistant"))
                else:
                    for agent in self.power_agent_map.values():
                        await agent.observe(Msg(name=sender, content=f"{sender} -> {recipient}: {content}", role="assistant"))
                if sender in self.power_agent_map:
                    await self.power_agent_map[sender].observe(Msg(name=sender, content=f"{sender} -> {recipient}: {content}", role="assistant"))
                else:
                    for agent in self.power_agent_map.values():
                        await agent.observe(Msg(name=sender, content=f"{sender} -> {recipient}: {content}", role="assistant"))

    async def _handle_order_phase(self, current_phase, phase_log):
        print(f"{Colors.OKBLUE}--- 开始书写命令 ---{Colors.ENDC}")
        possible_orders = self.game.get_all_possible_orders()
        
        async def process_agent_orders(power_name, power, agent):
            submitted_orders = []
            orderable_locations = self.game.get_orderable_locations(power_name)
            if not orderable_locations:
                return power_name, submitted_orders, []

            phase_type = current_phase[-1] if current_phase else ''
            phase_instruction = ''
            if phase_type == 'M': phase_instruction = self.prompts.get('order_instructions_movement_phase.txt', '')
            elif phase_type == 'R': phase_instruction = self.prompts.get('order_instructions_retreat_phase.txt', '')
            elif phase_type == 'A': phase_instruction = self.prompts.get('order_instructions_adjustment_phase.txt', '')
            else: phase_instruction = self.prompts.get('planning_instructions.txt', '')

            valid_orders_str = ""
            valid_orders_for_agent = []
            if orderable_locations:
                for loc in orderable_locations:
                    if loc in possible_orders and possible_orders[loc]:
                        orders = possible_orders[loc]
                        valid_orders_str += f"  Location {loc}: {orders}\n"
                        valid_orders_for_agent.extend(orders)
            
            order_prompt = (
                "---PHASE_INSTRUCTIONS---\n"
                f"{phase_instruction}\n\n"
                "IMPORTANT: Please output your orders as a list of strings. Example: ['A PAR - BUR', 'F BRE H']\n\n"
                "Valid Orders Reference:\n"
                f"{valid_orders_str}"
            )

            msg = Msg(name="Moderator", content=order_prompt, role="user")
            response_msg = await agent(msg)
            response_text = str(response_msg.content)
            phase_log["order_logs"][power_name] = response_text

            # Parse orders
            json_orders = []
            try:
                match = re.search(r'(\{[\s\S]*"orders"[\s\S]*\})', response_text)
                if match:
                    data = json.loads(match.group(1).replace('', ''))
                    if "orders" in data: json_orders = data["orders"]
            except Exception: pass

            if json_orders:
                for order in json_orders:
                    if order.strip() in valid_orders_for_agent:
                        submitted_orders.append(order.strip())
            
            if not submitted_orders:
                for order in valid_orders_for_agent:
                    if order in response_text: submitted_orders.append(order)
            
            translated_orders = [order_to_natural_language(o) for o in submitted_orders]
            print(f"{Colors.OKGREEN}{power_name} Orders: {translated_orders}{Colors.ENDC}")
            return power_name, submitted_orders, translated_orders

        tasks = []
        random_fallback_powers = []
        
        for power_name, power in self.game.powers.items():
            if power.is_eliminated(): continue
            if power_name in self.power_agent_map:
                tasks.append(process_agent_orders(power_name, power, self.power_agent_map[power_name]))
            else:
                random_fallback_powers.append(power_name)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for power_name, submitted_orders, translated_orders in results:
            if submitted_orders:
                self.game.set_orders(power_name, submitted_orders)
                phase_log["orders"][power_name] = submitted_orders
                await self._broadcast(f"{power_name} orders: {translated_orders}", sender="Moderator")
                for agent in self.power_agent_map.values():
                    await agent.observe(Msg(name="Moderator", content=f"{power_name} orders: {translated_orders}", role="assistant"))

        # Random fallback
        for power_name in random_fallback_powers:
            orderable_locations = self.game.get_orderable_locations(power_name)
            if not orderable_locations: continue
            submitted_orders = []
            for loc in orderable_locations:
                if loc in possible_orders and possible_orders[loc]:
                    submitted_orders.append(random.choice(possible_orders[loc]))
            if submitted_orders:
                self.game.set_orders(power_name, submitted_orders)
                phase_log["orders"][power_name] = submitted_orders


# ============================================================================
# Convenience Function (Backward Compatibility)
# ============================================================================

async def diplomacy_game(
    agents: List[AgentBase],
    max_phases: int = 10,
    map_name: str = 'standard',
    seed: int = 42,
    debug: bool = False,
    num_negotiation_rounds: int = 3,
    log_dir: str = None,
    state_manager: Any = None,
    observe_agent: AgentBase | None = None,
) -> Game:
    """
    Convenience function to run Diplomacy game.
    Wraps the DiplomacyGame class.
    """
    config = DiplomacyConfig(
        max_phases=max_phases,
        map_name=map_name,
        seed=seed,
        num_negotiation_rounds=num_negotiation_rounds,
        debug=debug
    )
    
    game = DiplomacyGame(
        agents=agents,
        config=config,
        log_dir=log_dir,
        state_manager=state_manager,
        language="zn" if os.getenv('LANGUAGE') == 'zn' else "en",
        observe_agent=observe_agent
    )
    
    return await game.run()
