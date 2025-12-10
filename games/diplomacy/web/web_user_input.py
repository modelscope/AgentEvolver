# -*- coding: utf-8 -*-
"""Web user input handler."""
import asyncio
import json
from typing import Any, Type
from pydantic import BaseModel

from agentscope.agent._user_input import UserInputBase, UserInputData
from agentscope.message import TextBlock

from games.diplomacy.web.game_state_manager import GameStateManager


class WebUserInput(UserInputBase):
    """Web-based user input handler that waits for input from frontend via WebSocket."""
    
    def __init__(self, state_manager: GameStateManager):
        """Initialize WebUserInput.
        
        Args:
            state_manager: GameStateManager instance for managing input queues
        """
        self.state_manager = state_manager
    
    async def __call__(
        self,
        agent_id: str,
        agent_name: str,
        *args: Any,
        structured_model: Type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> UserInputData:
        """Wait for and return user input from frontend.
        
        Args:
            agent_id: Agent identifier (UUID)
            agent_name: Agent name (e.g., "Player0")
            structured_model: Optional structured model for structured input
            
        Returns:
            UserInputData containing the user input
        """

        queue_key = agent_id  # Use agent_id as queue_key for mapping

        print(f"[WebUserInput] Waiting for input from {agent_name} (queue_key: {queue_key}, agent_id: {agent_id})")
        
        # Send request to frontend
        prompt = f"[{agent_name}] Please provide your input:"
        if structured_model is not None:
            prompt += f"\nStructured input required: {structured_model.model_json_schema()}"
        
        # Use queue_key (player ID) for the request so frontend can match it
        request_msg = self.state_manager.format_user_input_request(queue_key, prompt)
        await self.state_manager.broadcast_message(request_msg)
        print(f"[WebUserInput] Sent input request to frontend with queue_key: {queue_key}")
        
        # Wait for user input from queue using queue_key
        try:
            print(f"[WebUserInput] About to call get_user_input with queue_key: {queue_key}")
            print(f"[WebUserInput] State manager: {id(self.state_manager)}")
            print(f"[WebUserInput] Available queues: {list(self.state_manager.input_queues.keys())}")
            content = await self.state_manager.get_user_input(queue_key, timeout=None)
            # print(f"[WebUserInput] Received user input: {content[:50]}...")
            
            # For now, we only support text input
            # Structured input can be extended later
            structured_input = None
            if structured_model is not None:
                # Try to parse JSON from content
                try:
                    structured_input = json.loads(content)
                except json.JSONDecodeError:
                    # If not JSON, create a simple dict with the content
                    structured_input = {"content": content}
                    print(f"Warning: Failed to parse structured input as JSON for model {structured_model.__name__}")
            
            return UserInputData(
                blocks_input=[TextBlock(type="text", text=content)],
                structured_input=structured_input,
            )
        except Exception as e:
            # Return empty input on error
            return UserInputData(
                blocks_input=[TextBlock(type="text", text="")],
                structured_input=None,
            )

