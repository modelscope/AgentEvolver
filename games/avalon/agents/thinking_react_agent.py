# -*- coding: utf-8 -*-
"""A ReAct agent that thinks before speaking, with thinking content kept private."""
from typing import Type, Any, Literal
import re
import json

from pydantic import BaseModel

from agentscope.agent import ReActAgent
from agentscope.message import Msg, TextBlock
from agentscope.model import ChatModelBase


class ThinkingReActAgent(ReActAgent):
    """A ReAct agent that thinks before speaking.
    
    The thinking content is wrapped in <think>...</think>
    and is only stored in the agent's own memory, not broadcasted to other agents.
    """
    
    def __init__(
        self,
        name: str,
        sys_prompt: str,
        model: ChatModelBase,
        formatter,
        toolkit=None,
        memory=None,
        long_term_memory=None,
        long_term_memory_mode: Literal["agent_control", "static_control", "both"] = "both",
        enable_meta_tool: bool = False,
        parallel_tool_calls: bool = False,
        knowledge=None,
        enable_rewrite_query: bool = True,
        plan_notebook=None,
        print_hint_msg: bool = False,
        max_iters: int = 10,
        thinking_sys_prompt: str | None = None,
    ) -> None:
        """Initialize a ThinkingReActAgent.
        
        Args:
            thinking_sys_prompt: Optional system prompt for the thinking phase.
                If not provided, will use a default prompt that asks the agent
                to think first, then respond.
            Other args: Same as ReActAgent.
        """
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model=model,
            formatter=formatter,
            toolkit=toolkit,
            memory=memory,
            long_term_memory=long_term_memory,
            long_term_memory_mode=long_term_memory_mode,
            enable_meta_tool=enable_meta_tool,
            parallel_tool_calls=parallel_tool_calls,
            knowledge=knowledge,
            enable_rewrite_query=enable_rewrite_query,
            plan_notebook=plan_notebook,
            print_hint_msg=print_hint_msg,
            max_iters=max_iters,
        )
        
        # System prompt for thinking phase
        if thinking_sys_prompt is None:
            thinking_sys_prompt = (
                "Before you respond, think carefully about your response. "
                "Your thinking process should be wrapped in <think>...</think> tags. "
                "Then provide your actual response after the thinking section. "
                "Example format:\n"
                "<think>\n"
                "Your private thinking here...\n"
                "</think>\n"
                "Your actual response here."
            )
        
        # Append thinking instruction to system prompt permanently
        # No need to switch it back and forth in reply method
        self._sys_prompt = f"{self._sys_prompt}\n\n{thinking_sys_prompt}"
        
        # Store model call history: list of dicts with 'prompt' and 'response'
        self.model_call_history: list[dict[str, Any]] = []
    
    async def _reasoning(
        self,
        tool_choice: Literal["auto", "none", "any", "required"] | None = None,
    ) -> Msg:
        """Perform reasoning with thinking section.
        
        The complete message (with thinking) is stored in memory,
        but the returned message (for broadcast) excludes thinking content.
        """
                
        # Convert Msg objects into the required format of the model API
        # formatter.format returns list[dict[str, Any]] (messages format)
        prompt = await self.formatter.format(
            msgs=[
                Msg("system", self.sys_prompt, "system"),
                *await self.memory.get_memory(),
                # The hint messages to guide the agent's behavior, maybe empty
                *await self._reasoning_hint_msgs.get_memory(),
            ],
        )
        
        # Call parent reasoning to get the response
        # Parent's _reasoning will:
        # 1. Generate the complete response (potentially with thinking)
        # 2. Add the complete msg to memory in its finally block
        msg = await super()._reasoning(tool_choice)
        
        # Record model call history (prompt and response)
        if msg is not None:
            # Extract text content from response
            from games.avalon.utils import Parser
            response_content = Parser.extract_text_from_content(msg.content)
            
            # Store in history with prompt as messages list (prompt is already in messages format)
            call_record = {
                "prompt": prompt,  # prompt is already list[dict[str, Any]]
                "response": response_content,
                "response_msg": msg.to_dict() if hasattr(msg, 'to_dict') else {
                    "name": msg.name,
                    "content": response_content,
                    "role": msg.role,
                    "timestamp": str(msg.timestamp) if hasattr(msg, 'timestamp') else None,
                },
            }
            self.model_call_history.append(call_record)
        
        if msg is None:
            return msg
        
        # The parent _reasoning already added the complete msg (with thinking) to memory
        # We keep the memory as is - it contains the full model output
        
        # But we need to return a message without thinking for broadcast
        # Parse to get public message (for return, but memory keeps the full one)
        _, public_msg = self._separate_thinking_and_response(msg)
        
        # Create a new message object for return (without thinking, but same ID)
        # This doesn't affect memory, which still has the full message
        return_msg = Msg(
            name=msg.name,
            content=public_msg.content,
            role=msg.role,
            metadata=msg.metadata,
        )
        return_msg.id = msg.id
        return_msg.timestamp = msg.timestamp
        
        return return_msg
    
    def _separate_thinking_and_response(
        self,
        msg: Msg,
    ) -> tuple[Msg | None, Msg]:
        """Separate thinking content from public response.
        
        Args:
            msg: The original message that may contain thinking section.
            
        Returns:
            A tuple of (thinking_msg, public_msg):
            - thinking_msg: Message containing only thinking content (or None)
            - public_msg: Message containing only public response content
        """
        # Get text content
        text_content = msg.get_text_content()
        
        # Pattern to match <think>...</think>
        pattern = r'<think>(.*?)</think>'
        matches = re.findall(pattern, text_content, re.DOTALL)
        
        thinking_content = None
        if matches:
            # Extract thinking content
            thinking_content = matches[0].strip()
            
            # Remove thinking section from public content
            public_content = re.sub(pattern, '', text_content, flags=re.DOTALL).strip()
        else:
            # No thinking section found, all content is public
            public_content = text_content
        
        # Create thinking message if thinking content exists
        thinking_msg = None
        if thinking_content:
            thinking_msg = Msg(
                name=self.name,
                content=[
                    TextBlock(
                        type="text",
                        text=f"<think>\n{thinking_content}\n</think>",
                    ),
                ],
                role="assistant",
            )
        
        # Create public message (without thinking)
        # Reconstruct content blocks
        public_blocks = []
        if isinstance(msg.content, str):
            public_blocks = [
                TextBlock(type="text", text=public_content),
            ]
        elif isinstance(msg.content, list):
            # Check if there are non-text blocks (like tool_use)
            has_non_text = any(
                block.get("type") != "text"
                for block in msg.content
            )
            
            if has_non_text:
                # Keep non-text blocks, update text blocks
                for block in msg.content:
                    if block.get("type") == "text":
                        # Remove thinking from text blocks
                        block_text = block.get("text", "")
                        if "<think>" in block_text:
                            cleaned_text = re.sub(
                                pattern,
                                '',
                                block_text,
                                flags=re.DOTALL,
                            ).strip()
                            if cleaned_text:
                                public_blocks.append(
                                    TextBlock(type="text", text=cleaned_text),
                                )
                        else:
                            public_blocks.append(block)
                    else:
                        public_blocks.append(block)
            else:
                # All text blocks, use cleaned content
                if public_content:
                    public_blocks = [
                        TextBlock(type="text", text=public_content),
                    ]
        
        public_msg = Msg(
            name=msg.name,
            content=public_blocks or msg.content,
            role=msg.role,
            metadata=msg.metadata,
        )
        # Use same ID as original message
        public_msg.id = msg.id
        public_msg.timestamp = msg.timestamp
        
        return thinking_msg, public_msg
    
    # Note: We don't need to override _broadcast_to_subscribers
    # because reply() already returns a message without thinking content.
    # The _broadcast_to_subscribers will receive the public message directly.
    
    # Note: We don't need to override reply() anymore.
    # The thinking instruction is already added to system prompt in __init__.
    # The parent's reply() and our _reasoning() override handle everything.

