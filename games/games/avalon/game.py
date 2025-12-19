# -*- coding: utf-8 -*-
"""Avalon game implemented by agentscope."""
from typing import Any

from agentscope.agent import AgentBase
from agentscope.message import Msg
from agentscope.pipeline import MsgHub, fanout_pipeline

from loguru import logger

from games.games.avalon.engine import AvalonGameEnvironment, AvalonBasicConfig
from games.games.avalon.utils import Parser, GameLogger, LanguageFormatter
from games.agents.echo_agent import EchoAgent


class AvalonGame:
    """Main Avalon game class that integrates all game functionality."""
    
    def __init__(
        self,
        agents: list[AgentBase],
        config: AvalonBasicConfig,
        log_dir: str | None = None,
        language: str = "en",
        observe_agent: AgentBase | None = None,
        state_manager: Any = None,
        preset_roles: list[tuple[int, str, bool]] | None = None,
        timestamp: str | None = None,
    ):
        """Initialize Avalon game.
        
        Args:
            agents: List of agents (5-10 players). Can be ReActAgent, ThinkingReActAgent, or UserAgent.
            config: Game configuration.
            log_dir: Directory to save game logs. If None, logs are not saved.
            language: Language for prompts. "en" for English, "zh" or "cn" for Chinese.
            observe_agent: Optional observer agent to add to all hubs. Default is None.
            state_manager: Optional state manager for web mode to check stop flag.
            preset_roles: Optional list of preset roles as tuples (role_id, role_name, is_good).
                If provided, uses these roles instead of random assignment.
            timestamp: Optional timestamp string for log directory naming. If None, generates a new one.
        """
        self.agents = agents
        self.config = config
        self.log_dir = log_dir
        self.language = language
        self.observe_agent = observe_agent
        self.state_manager = state_manager
        
        # Initialize utilities
        self.localizer = LanguageFormatter(language)
        self.parser = Parser()
        self.game_logger = GameLogger()
        
        # Initialize moderator
        self.moderator = EchoAgent()
        self.moderator.set_console_output_enabled(True)
        
        # Import prompts based on language
        if self.localizer.is_zh:
            from games.games.avalon.prompt import ChinesePrompts as Prompts
        else:
            from games.games.avalon.prompt import EnglishPrompts as Prompts
        self.Prompts = Prompts
        
        # Initialize game environment with preset roles if provided
        if preset_roles is not None:
            # Use preset roles - create environment with presets
            # Extract role names (they should already be in correct format from get_roles())
            role_names = [role_name for _, role_name, _ in preset_roles]
            import numpy as np
            quest_leader = np.random.randint(0, config.num_players - 1)
            presets = {
                'num_players': config.num_players,
                'quest_leader': quest_leader,
                'role_names': role_names,
            }
            self.env = AvalonGameEnvironment.from_presets(presets)
            # Fix: from_presets uses cls variables, so we need to set instance variables
            # Convert preset_roles to the format needed by env (role_ids array)
            role_ids = [role_id for role_id, _, _ in preset_roles]
            is_good_list = [is_good for _, _, is_good in preset_roles]
            import numpy as np
            self.env.roles = np.array(role_ids)
            self.env.is_good = np.array(is_good_list)
            self.env.quest_leader = quest_leader
            # Use the preset roles directly
            self.roles = preset_roles
        else:
            # Use default random role assignment
            self.env = AvalonGameEnvironment(config)
            self.roles = self.env.get_roles()
        
        # Initialize game log
        self.game_logger.create_game_log_dir(log_dir, timestamp)
        self.game_logger.initialize_game_log(self.roles, config.num_players)
        
        assert len(agents) == config.num_players, f"The Avalon game needs exactly {config.num_players} players."
    
    def _get_hub_participants(self) -> list[AgentBase]:
        """Get participants list for hub, including observe_agent if present."""
        participants = self.agents.copy()
        if self.observe_agent is not None:
            participants.append(self.observe_agent)
        return participants
    
    async def run(self) -> bool:
        """Run the Avalon game.
        
        Returns:
            True if good wins, False otherwise.
        """
        # Broadcast game begin message and system prompt
        async with MsgHub(participants=self._get_hub_participants()) as greeting_hub:
            # Format system prompt using localizer
            system_prompt_content = self.localizer.format_system_prompt(self.config, self.Prompts)
            system_prompt_msg = await self.moderator(system_prompt_content)
            await greeting_hub.broadcast(system_prompt_msg)
            
            new_game_msg = await self.moderator(
                self.Prompts.to_all_new_game.format(self.localizer.format_agents_names(self.agents))
            )
            await greeting_hub.broadcast(new_game_msg)

        # Assign roles to agents
        await self._assign_roles_to_agents()
        
        # Broadcast roles to frontend for observe mode
        if self.state_manager:
            # Convert roles to serializable format (convert numpy int64 to Python int)
            roles_data = [
                {
                    "role_id": int(role_id),
                    "role_name": str(role_name),
                    "is_good": bool(is_good)
                }
                for role_id, role_name, is_good in self.roles
            ]
            self.state_manager.update_game_state(roles=roles_data)
            await self.state_manager.broadcast_message(self.state_manager.format_game_state())

        # Main game loop
        game_stopped = False
        while not self.env.done:
            # Check if game should stop (for web mode)
            if self.state_manager and self.state_manager.should_stop:
                logger.info("Game stopped by user request")
                game_stopped = True
                # Mark environment as done to exit loop
                self.env.done = True
                break
            
            phase, _ = self.env.get_phase()
            leader = self.env.get_quest_leader()
            mission_id = self.env.turn
            round_id = self.env.round

            # Update and broadcast game state for web frontend
            if self.state_manager:
                self.state_manager.update_game_state(
                    phase=phase,
                    mission_id=mission_id,
                    round_id=round_id,
                    leader=leader,
                )
                await self.state_manager.broadcast_message(self.state_manager.format_game_state())

            async with MsgHub(participants=self._get_hub_participants(), enable_auto_broadcast=False, name="all_players") as all_players_hub:
                # Check again inside the hub context
                if self.state_manager and self.state_manager.should_stop:
                    logger.info("Game stopped by user request")
                    game_stopped = True
                    self.env.done = True
                    break
                    
                if phase == 0:
                    await self._handle_team_selection_phase(
                        all_players_hub, mission_id, round_id, leader
                    )
                elif phase == 1:
                    await self._handle_team_voting_phase(all_players_hub)
                elif phase == 2:
                    await self._handle_quest_voting_phase(all_players_hub, mission_id)
                elif phase == 3:
                    await self._handle_assassination_phase(all_players_hub)

        # Only broadcast final result if game completed normally (not stopped)
        if not game_stopped:
            # Game over - broadcast final result
            async with MsgHub(participants=self._get_hub_participants()) as end_hub:
                end_message = self.localizer.format_game_end_message(
                    self.env.good_victory,
                    self.roles,
                    self.Prompts
                )
                end_msg = await self.moderator(end_message)
                await end_hub.broadcast(end_msg)

            logger.info(f"Game finished. Good wins: {self.env.good_victory}, Quest results: {self.env.quest_results}")
            
            # Save game log and agent memories
            await self.game_logger.save_game_logs(self.agents, self.env, self.roles)
            
            return self.env.good_victory
        else:
            # Game was stopped, return None to indicate it was stopped
            logger.info("Game was stopped by user")
            return None
    
    async def _assign_roles_to_agents(self) -> None:
        """Assign roles to agents and inform them of their roles and visibility."""
        MERLIN_ROLE_ID = 0
        EVIL_SIDE = 0
        
        for i, (role_id, role_name, side) in enumerate(self.roles):
            if hasattr(self.agents[i], 'model') and self.agents[i].model is not None:
                logger.info(f"Assigning role to agent {i} {self.agents[i].model.model_name}: {role_name}, {side}")
            else:
                logger.info(f"Assigning role to agent {i} Human: {role_name}, {side}")

            agent = self.agents[i]
            localized_role_name = self.localizer.format_role_name(role_name)
            side_name = self.localizer.format_side_name(side)
            localized_agent_name = self.localizer.format_player_name(agent.name)
            
            # Build visibility info
            if role_id == MERLIN_ROLE_ID or side == EVIL_SIDE:
                sides_info = self.localizer.format_sides_info(self.roles)
                additional_info = self.Prompts.to_agent_role_with_visibility.format(sides_info=", ".join(sides_info))
            else:
                additional_info = self.Prompts.to_agent_role_no_visibility
            
            role_info = self.Prompts.to_agent_role_assignment.format(
                agent_name=localized_agent_name,
                role_name=localized_role_name,
                side_name=side_name,
                additional_info=additional_info,
            )
            
            # Send role info to agent (private, not broadcasted)
            role_msg = Msg(
                name="Moderator",
                content=role_info,
                role="assistant",
            )
            await agent.observe(role_msg)
    
    async def _handle_team_selection_phase(
        self,
        all_players_hub: MsgHub,
        mission_id: int,
        round_id: int,
        leader: int,
    ) -> None:
        """Handle Team Selection Phase."""
        # Add mission to log
        self.game_logger.add_mission(mission_id, round_id, leader)
        
        # Broadcast phase and discussion prompt
        phase_msg = await self.moderator(self.Prompts.to_all_team_selection_discuss.format(
            mission_id=mission_id,
            round_id=round_id,
            leader_id=leader,
            team_size=self.env.get_team_size(),
        ))
        await all_players_hub.broadcast(phase_msg)

        # Discussion: leader speaks first, then others
        leader_agent = self.agents[leader]
        all_players_hub.set_auto_broadcast(True)
        discussion_msgs = []
        
        # Leader speaks
        leader_msg = await leader_agent()
        discussion_msgs.append(leader_msg)
        
        # Others speak in order
        for i in range(1, self.config.num_players):
            agent = self.agents[(leader + i) % self.config.num_players]
            msg = await agent()
            discussion_msgs.append(msg)
        
        all_players_hub.set_auto_broadcast(False)
        
        # Add discussion to log
        self.game_logger.add_discussion_messages([msg.to_dict() for msg in discussion_msgs])

        # Leader proposes team
        propose_prompt = await self.moderator(self.Prompts.to_leader_propose_team.format(
            mission_id=mission_id,
            team_size=self.env.get_team_size(),
            max_player_id=self.config.num_players - 1,
        ))
        team_response = await leader_agent(propose_prompt)
        team = self.parser.parse_team_from_response(team_response.content)
        
        # Normalize team size
        # TODO[1202]: move this to utils.py
        team = list(set(team))[:self.env.get_team_size()]
        if len(team) < self.env.get_team_size():
            team.extend([i for i in range(self.config.num_players) if i not in team][:self.env.get_team_size() - len(team)])
        
        self.env.choose_quest_team(team=frozenset(team), leader=leader)
        
        # Add team proposal to log
        self.game_logger.add_team_proposal(list(team))
    
    async def _handle_team_voting_phase(
        self,
        all_players_hub: MsgHub,
    ) -> None:
        """Handle Team Voting Phase."""
        current_team = self.env.get_current_quest_team()
        
        # Send voting prompt to all agents (private)
        vote_prompt = await self.moderator(self.Prompts.to_all_team_vote.format(team=list(current_team)))

        # Collect votes - vote_prompt is sent to all agents
        msgs_vote = await fanout_pipeline(self.agents, msg=[vote_prompt], enable_gather=True)
        votes = [self.parser.parse_vote_from_response(msg.content) for msg in msgs_vote]
        outcome = self.env.gather_team_votes(votes)
        
        # Format and broadcast results
        approved = bool(outcome[2])
        votes_detail, result_text, outcome_text = self.localizer.format_vote_details(votes, approved)
        
        result_msg = await self.moderator(self.Prompts.to_all_team_vote_result.format(
            result=result_text,
            team=list(current_team),
            outcome=outcome_text,
            votes_detail=votes_detail,
        ))
        await all_players_hub.broadcast([result_msg])
        
        # Add team voting to log
        self.game_logger.add_team_voting(list(current_team), votes, approved)
    
    async def _handle_quest_voting_phase(
        self,
        all_players_hub: MsgHub,
        mission_id: int,
    ) -> None:
        """Handle Quest Voting Phase."""
        current_team = self.env.get_current_quest_team()
        team_agents = [self.agents[i] for i in current_team]
        
        # Send voting prompt only to team agents (private)
        vote_prompt = await self.moderator(self.Prompts.to_all_quest_vote.format(team=list(current_team)))

        # Collect votes (private) - vote_prompt is sent only to team_agents
        msgs_vote = await fanout_pipeline(team_agents, msg=[vote_prompt], enable_gather=True)
        votes = [self.parser.parse_vote_from_response(msg.content) for msg in msgs_vote]
        outcome = self.env.gather_quest_votes(votes)
        
        # Broadcast result only
        result_msg = await self.moderator(self.Prompts.to_all_quest_result.format(
            mission_id=mission_id,
            outcome="succeeded" if outcome[2] else "failed",
            team=list(current_team),
            num_fails=outcome[3],
        ))
        await all_players_hub.broadcast(result_msg)
        
        # Add quest voting to log
        self.game_logger.add_quest_voting(
            list(current_team),
            votes,
            int(outcome[3]),
            bool(outcome[2])
        )
    
    async def _handle_assassination_phase(
        self,
        all_players_hub: MsgHub,
    ) -> None:
        """Handle Assassination Phase."""
        # Broadcast phase
        assassination_msg = await self.moderator(self.Prompts.to_all_assassination)
        await all_players_hub.broadcast(assassination_msg)

        # Assassin chooses target
        assassin_id = self.env.get_assassin()
        assassinate_prompt = await self.moderator(
            self.Prompts.to_assassin_choose.format(max_player_id=self.config.num_players - 1)
        )
        target_response = await self.agents[assassin_id](assassinate_prompt)
        target = self.parser.parse_player_id_from_response(target_response.content, self.config.num_players - 1)
        _, _, good_wins = self.env.choose_assassination_target(assassin_id, target)
        
        # Broadcast result
        assassin_name = self.localizer.format_player_id(assassin_id)
        target_name = self.localizer.format_player_id(target)
        result_text = self.Prompts.to_all_good_wins if good_wins else self.Prompts.to_all_evil_wins
        
        if self.localizer.is_zh:
            result_msg = await self.moderator(f"刺客{assassin_name} 选择刺杀{target_name}。{result_text}")
        else:
            result_msg = await self.moderator(f"Assassin {assassin_name} has chosen to assassinate {target_name}. {result_text}")
        await all_players_hub.broadcast(result_msg)
        
        # Add assassination to log
        self.game_logger.add_assassination(assassin_id, target, bool(good_wins))


# ============================================================================
# Convenience Function
# ============================================================================

async def avalon_game(
    agents: list[AgentBase],
    config: AvalonBasicConfig,
    log_dir: str | None = None,
    language: str = "en",
    web_mode: str | None = None,
    web_observe_agent: AgentBase | None = None,
    state_manager: Any = None,
    preset_roles: list[tuple[int, str, bool]] | None = None,  # added gpt
) -> bool:
    """Convenience function to run Avalon game.
    
    This is a wrapper around AvalonGame class for backward compatibility.
    
    Args:
        agents: List of agents (5-10 players). Can be ReActAgent, ThinkingReActAgent, or UserAgent.
        config: Game configuration.
        log_dir: Directory to save game logs. If None, logs are not saved.
        language: Language for prompts. "en" for English, "zh" or "cn" for Chinese.
        web_mode: Web mode ("observe" or "participate"). If None, runs in normal mode.
        web_observe_agent: Observer agent for web observe mode. Only used when web_mode="observe".
        state_manager: Optional state manager for web mode to check stop flag.
    Returns:
        True if good wins, False otherwise.
    """
    # Create AvalonGame instance
    game = AvalonGame(
        agents=agents,
        config=config,
        log_dir=log_dir,
        language=language,
        observe_agent=web_observe_agent if web_mode == "observe" else None,
        state_manager=state_manager,
        preset_roles=preset_roles,  # added gpt
    )
    
    # Run the game
    return await game.run()
