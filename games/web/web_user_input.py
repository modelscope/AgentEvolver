# -*- coding: utf-8 -*-
"""Unified web user input handler."""
import json
from typing import Any, Type
from pydantic import BaseModel

from agentscope.agent._user_input import UserInputBase, UserInputData
from agentscope.message import TextBlock

from games.web.game_state_manager import GameStateManager  #add gpt unified gsm


class WebUserInput(UserInputBase):
    """Web-based user input handler that waits for input from frontend via WebSocket."""
    
    def __init__(self, state_manager: GameStateManager):
        self.state_manager = state_manager  #add gpt reuse unified gsm
    
    async def __call__(
        self,
        agent_id: str,
        agent_name: str,
        *args: Any,
        structured_model: Type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> UserInputData:
        queue_key = agent_id
        prompt = f"[{agent_name}] Please provide your input:"
        if structured_model is not None:
            prompt += f"\nStructured input required: {structured_model.model_json_schema()}"
        
        request_msg = self.state_manager.format_user_input_request(queue_key, prompt)
        await self.state_manager.broadcast_message(request_msg)
        
        try:
            content = await self.state_manager.get_user_input(queue_key, timeout=None)
            structured_input = None
            if structured_model is not None:
                try:
                    structured_input = json.loads(content)
                except json.JSONDecodeError:
                    structured_input = {"content": content}
            
            return UserInputData(
                blocks_input=[TextBlock(type="text", text=content)],
                structured_input=structured_input,
            )
        except Exception:
            return UserInputData(
                blocks_input=[TextBlock(type="text", text="")],
                structured_input=None,
            )

