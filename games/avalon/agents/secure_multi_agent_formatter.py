# -*- coding: utf-8 -*-
"""Secure multi-agent formatter with thinking block support and anti-forgery protection."""
import re
from typing import Any

from agentscope.formatter import OpenAIMultiAgentFormatter
from agentscope.message import Msg, ThinkingBlock


class SecureMultiAgentFormatter(OpenAIMultiAgentFormatter):
    """
    A secure multi-agent formatter that extends OpenAIMultiAgentFormatter with:
    1. Support for ThinkingBlock in messages (to see past thinking)
    2. Anti-forgery protection: uses special format tags to prevent agents from
       forging other agents' outputs, and removes any such format from agent outputs.
    
    The formatter uses `<agent:name>` and `</agent:name>` tags to mark each agent's
    output in conversation history. Any content containing these tags in agent outputs
    will be removed to prevent forgery.
    """
    
    # Agent output format pattern: <agent:name>content</agent:name>
    AGENT_TAG_PATTERN = re.compile(r'<agent:([^>]+)>(.*?)</agent:\1>', re.DOTALL)
    
    def __init__(
        self,
        conversation_history_prompt: str = (
            "# Conversation History\n"
            "The content between <history></history> tags contains "
            "your conversation history\n"
        ),
        promote_tool_result_images: bool = False,
        token_counter=None,
        max_tokens: int | None = None,
        preserved_agent_names: list[str] | str | None = None,
    ) -> None:
        """Initialize the SecureMultiAgentFormatter.
        
        Args:
            conversation_history_prompt: Prompt for conversation history section.
            promote_tool_result_images: Whether to promote images from tool results.
            token_counter: Token counter instance.
            max_tokens: Maximum tokens allowed.
            preserved_agent_names: Agent names whose messages should be preserved during truncation.
                Can be a single string, a list of strings, or None (no preservation).
        """
        super().__init__(
            conversation_history_prompt=conversation_history_prompt,
            promote_tool_result_images=promote_tool_result_images,
            token_counter=token_counter,
            max_tokens=max_tokens,
        )
        
        # Normalize preserved_agent_names to a set for efficient lookup
        if preserved_agent_names is None:
            self.preserved_agent_names = set()
        elif isinstance(preserved_agent_names, str):
            self.preserved_agent_names = {preserved_agent_names}
        else:
            self.preserved_agent_names = set(preserved_agent_names)
    
    async def _count(self, msgs: list[dict[str, Any]]) -> int | None:
        """
        Count tokens in formatted messages, converting OpenAI format to string format
        for HuggingFaceTokenCounter compatibility.
        
        Args:
            msgs: List of formatted message dictionaries.
            
        Returns:
            Token count or None if token_counter is not available.
        """
        if self.token_counter is None:
            return None
        
        # Convert OpenAI format (content as list) to string format for HuggingFaceTokenCounter
        reformatted_msgs = []
        for msg in msgs:
            reformatted_msg = {**msg}
            content = msg.get("content", "")
            
            # If content is a list (OpenAI format), convert to string
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text" and "text" in item:
                            text_parts.append(item["text"])
                        # For other types (image, audio, etc.), we skip them for token counting
                if text_parts:
                    reformatted_msg["content"] = "\n".join(text_parts)
                else:
                    # If no text content, use empty string
                    reformatted_msg["content"] = ""
            # If content is already a string, use it as is
            elif isinstance(content, str):
                reformatted_msg["content"] = content
            else:
                # Fallback: convert to string
                reformatted_msg["content"] = str(content) if content else ""
            
            reformatted_msgs.append(reformatted_msg)
        
        return await self.token_counter.count(reformatted_msgs)
    
    def _remove_agent_tags(self, text: str) -> str:
        """
        Remove all agent tag patterns from text to prevent forgery.
        
        Args:
            text: Input text that may contain agent tags.
            
        Returns:
            Text with all agent tags removed.
        """
        # Remove all <agent:name>...</agent:name> patterns
        cleaned_text = self.AGENT_TAG_PATTERN.sub('', text)
        # Also remove any incomplete tags (opening or closing)
        cleaned_text = re.sub(r'<agent:[^>]*>', '', cleaned_text)
        cleaned_text = re.sub(r'</agent:[^>]*>', '', cleaned_text)
        return cleaned_text.strip()
    
    async def _truncate(self, msgs: list[Msg]) -> list[Msg]:
        """
        Truncate messages while preserving messages from specified agent names.
        
        This method removes the oldest message that is not from preserved agents,
        while keeping all messages from preserved agents and the system message.
        The parent class's format() method will call this repeatedly until
        the token count is within limits.
        
        Args:
            msgs: List of messages to truncate.
            
        Returns:
            Truncated list of messages with preserved agent messages kept.
        """
        if not msgs:
            return msgs
        
        # Separate messages into categories
        system_msg = None
        preserved_msgs = []  # Messages from preserved agents
        other_msgs = []  # Other messages
        
        # Handle system message
        start_index = 0
        if len(msgs) > 0 and msgs[0].role == "system":
            system_msg = msgs[0]
            start_index = 1
        
        # Categorize remaining messages
        for i in range(start_index, len(msgs)):
            msg = msgs[i]
            # Check if message is from a preserved agent
            msg_name = getattr(msg, "name", None) or ""
            if msg_name in self.preserved_agent_names:
                preserved_msgs.append((i, msg))
            else:
                other_msgs.append((i, msg))
        
        # If no other messages to remove, return original
        if not other_msgs:
            return msgs
        
        # Remove the oldest non-preserved message
        # Sort by index to maintain order, then remove the first (oldest)
        other_msgs.sort(key=lambda x: x[0])
        oldest_other_index = other_msgs[0][0]
        
        # Reconstruct message list, excluding the oldest non-preserved message
        truncated_msgs = []
        if system_msg is not None:
            truncated_msgs.append(system_msg)
        
        # Add all messages except the one we're removing
        for i in range(start_index, len(msgs)):
            if i != oldest_other_index:
                truncated_msgs.append(msgs[i])
        
        return truncated_msgs
    
    async def _format_agent_message(
        self,
        msgs: list[Msg],
        is_first: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Format agent messages with thinking block support and anti-forgery protection.
        
        Args:
            msgs: List of messages to format.
            is_first: Whether this is the first agent message.
            
        Returns:
            Formatted messages for OpenAI API.
        """
        if is_first:
            conversation_history_prompt = self.conversation_history_prompt
        else:
            conversation_history_prompt = ""

        # Format into required OpenAI format
        formatted_msgs: list[dict] = []
        accumulated_text = []

        for msg in msgs:
            # Collect thinking and text blocks separately to ensure order
            thinking_content = None
            text_content = None
            
            for block in msg.get_content_blocks():
                if block["type"] == "thinking":
                    # Collect thinking content first
                    thinking_content = block.get("thinking", "")
                elif block["type"] == "text":
                    # Remove agent tags from text to prevent forgery
                    cleaned_text = self._remove_agent_tags(block["text"])
                    if cleaned_text:
                        text_content = cleaned_text
            
            # Build message content: thinking first, then text
            msg_parts = []
            if thinking_content:
                msg_parts.append(f"<think>{thinking_content}</think>")
            if text_content:
                msg_parts.append(text_content)
            
            # Combine all parts of the same message into one line
            if msg_parts:
                combined_content = " ".join(msg_parts)
                accumulated_text.append(f"<agent:{msg.name}>{combined_content}</agent:{msg.name}>")

        # Build conversation history text
        if accumulated_text:
            conversation_text = "\n".join(accumulated_text)
            conversation_text = (
                conversation_history_prompt
                + "<history>\n"
                + conversation_text
                + "\n</history>"
            )
        else:
            conversation_text = conversation_history_prompt + "<history>\n</history>"

        # Build content list
        content_list: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": conversation_text,
            }
        ]

        user_message = {
            "role": "user",
            "content": content_list,
        }

        formatted_msgs.append(user_message)

        return formatted_msgs

