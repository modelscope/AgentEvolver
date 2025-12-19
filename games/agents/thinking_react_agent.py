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
        # thinking_tag_start_end: tuple[str, str] = ("<privacy_think>", "</privacy_think>"),
        thinking_tag_start_end: tuple[str, str] = ("<think>", "</think>"),
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
        thinking_tag_start, thinking_tag_end = thinking_tag_start_end
        self.thinking_tag_start = thinking_tag_start
        self.thinking_tag_end = thinking_tag_end
        if thinking_sys_prompt is None:
            thinking_sys_prompt = (
                "Before you respond, think carefully about your response. "
                f"Your thinking process should be wrapped in {thinking_tag_start}...{thinking_tag_end} tags. "
                "Then provide your actual response after the thinking section. "
                "Example format:\n"
                f"{thinking_tag_start}\n"
                "Your private thinking here...\n"
                f"{thinking_tag_end}\n"
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
            response_content = msg.get_text_content()
            
            # Extract tokens from msg metadata if available (from AgentscopeModelWrapper)
            # The tokens are stored in ChatResponse.metadata by AgentscopeModelWrapper
            # Agentscope typically copies ChatResponse.metadata to Msg.metadata
            tokens = None
            if hasattr(msg, 'metadata') and msg.metadata:
                # Check if metadata contains tokens (from ChatResponse.metadata)
                if isinstance(msg.metadata, dict):
                    # Check for 'tokens' key (token_ids list)
                    if 'tokens' in msg.metadata:
                        tokens = msg.metadata['tokens']
                    # Also check for 'original_tokens' (token objects with token_id attribute)
                    elif 'original_tokens' in msg.metadata:
                        original_tokens = msg.metadata['original_tokens']
                        if original_tokens:
                            tokens = [t.token_id if hasattr(t, 'token_id') else t for t in original_tokens]
            
            # Store in history with prompt as messages list (prompt is already in messages format)
            call_record = {
                "prompt": prompt,  # prompt is already list[dict[str, Any]]
                "response": response_content,
                "response_msg": msg.to_dict() if hasattr(msg, 'to_dict') else {},
            }
            # Add tokens if available (for training consistency)
            if tokens is not None:
                call_record["tokens"] = tokens
            self.model_call_history.append(call_record)
        
        if msg is None:
            return msg
        
        # The parent _reasoning already added the complete msg (with thinking) to memory
        # We keep the memory as is - it contains the full model output
        
        # But we need to return a message without thinking for broadcast
        # Parse to get public message (for return, but memory keeps the full one)
        return self._separate_thinking_and_response(msg)
    
    def _separate_thinking_and_response(
        self,
        msg: Msg,
    ) -> Msg:
        """Remove thinking content from message and return public message only.
        
        Args:
            msg: The original message that may contain thinking section.
            
        Returns:
            Message containing only public response content (without thinking).
        """
        # Pattern to match <think>...</think> in text
        pattern = f'{self.thinking_tag_start}(.*?){self.thinking_tag_end}'
        public_blocks = []
        
        # Handle content as list of blocks (typical case for OpenAIChatModel)
        if isinstance(msg.content, list):
            for block in msg.content:
                block_type = block.get("type")
                
                if block_type == "thinking":
                    # Skip thinking blocks
                    continue
                    
                elif block_type == "text":
                    # TextBlock - check if it contains <think> tags
                    text_content = block.get("text", "")
                    # Remove thinking section from text
                    cleaned_text = re.sub(pattern, '', text_content, flags=re.DOTALL).strip()
                    if cleaned_text:
                        public_blocks.append(
                            TextBlock(type="text", text=cleaned_text),
                        )
                else:
                    # Other block types (tool_use, image, audio, etc.) - keep as public
                    public_blocks.append(block)
                    
        elif isinstance(msg.content, str):
            # Handle content as string (fallback case)
            text_content = msg.content
            # Remove thinking section from text
            public_content = re.sub(pattern, '', text_content, flags=re.DOTALL).strip()
            if public_content:
                public_blocks.append(
                    TextBlock(type="text", text=public_content),
                )
        
        # Create public message (without thinking)
        # Use empty list/string if no public content remains
        final_content = public_blocks if public_blocks else ([] if isinstance(msg.content, list) else "")
        
        public_msg = Msg(
            name=msg.name,
            content=final_content,
            role=msg.role,
            metadata=msg.metadata,
        )
        # Use same ID as original message
        public_msg.id = msg.id
        public_msg.timestamp = msg.timestamp
        
        return public_msg
    
    # Note: We don't need to override _broadcast_to_subscribers
    # because reply() already returns a message without thinking content.
    # The _broadcast_to_subscribers will receive the public message directly.
    
    # Note: We don't need to override reply() anymore.
    # The thinking instruction is already added to system prompt in __init__.
    # The parent's reply() and our _reasoning() override handle everything.

