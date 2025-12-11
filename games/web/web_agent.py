# -*- coding: utf-8 -*-
"""Unified web agents for Avalon + Diplomacy."""
from typing import Any
from agentscope.agent import AgentBase, UserAgent
from agentscope.message import Msg

from games.web.game_state_manager import GameStateManager  #add gpt unified gsm
from games.web.web_user_input import WebUserInput  #add gpt unified user input
from games.avalon.utils import Parser as AvalonParser  #add gpt import Avalon parser for content extraction


class WebUserAgent(UserAgent):
    """Web-based UserAgent that syncs observations to frontend and uses WebUserInput."""
    
    def __init__(self, name: str, state_manager: GameStateManager):
        super().__init__(name=name)
        self.state_manager = state_manager
        self.agent_id = self.id
        web_input = WebUserInput(state_manager)  #add gpt unified user input for both games
        self.override_instance_input_method(web_input)
    
    async def observe(self, msg: Msg | list[Msg] | None) -> None:
        await super().observe(msg)
        if self.state_manager.mode != "participate" or msg is None:
            return
        messages = msg if isinstance(msg, list) else [msg]
        for m in messages:
            if isinstance(m, Msg):
                content = AvalonParser.extract_text_from_content(m.content)  #add gpt extract text from complex content structure
                sender = m.name
                role = m.role
            else:
                content = str(m)
                sender = "System"
                role = "assistant"
            await self.state_manager.broadcast_message(
                self.state_manager.format_message(sender=sender, content=content, role=role)
            )
    
    async def reply(self, msg: Msg | list[Msg] | None = None, structured_model: Any = None) -> Msg:
        if msg is not None:
            await self.observe(msg)
        return await super().reply(msg=msg, structured_model=structured_model)


class ObserveAgent(AgentBase):
    """Observer agent that forwards all messages to frontend."""
    
    def __init__(self, name: str, state_manager: GameStateManager, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.state_manager = state_manager
    
    async def observe(self, msg: Msg | list[Msg] | None) -> None:
        messages = msg if isinstance(msg, list) else [msg]
        for m in messages:
            if isinstance(m, Msg):
                content = AvalonParser.extract_text_from_content(m.content)  #add gpt extract text from complex content structure
                sender = m.name
            else:
                content = str(m)
                sender = "Unknown"
            await self.state_manager.broadcast_message(
                {"type": "message", "sender": sender, "content": content, "role": "assistant"}
            )
    
    def reply(self, x: dict = None) -> Msg:
        return Msg(self.name, content="", role="assistant")  #add gpt observer dummy reply

