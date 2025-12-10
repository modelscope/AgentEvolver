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
    ) -> None:
        """Initialize the SecureMultiAgentFormatter.
        
        Args:
            conversation_history_prompt: Prompt for conversation history section.
            promote_tool_result_images: Whether to promote images from tool results.
            token_counter: Token counter instance.
            max_tokens: Maximum tokens allowed.
        """
        super().__init__(
            conversation_history_prompt=conversation_history_prompt,
            promote_tool_result_images=promote_tool_result_images,
            token_counter=token_counter,
            max_tokens=max_tokens,
        )
    
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
    
    def _format_agent_message(
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

