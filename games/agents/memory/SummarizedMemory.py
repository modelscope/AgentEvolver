# -*- coding: utf-8 -*-
"""The dialogue memory class with automatic summarization."""

import os
import logging
import yaml
from pathlib import Path
from typing import Union, Iterable, Any, Optional, Dict, TYPE_CHECKING

from agentscope.memory import MemoryBase
from agentscope.message import Msg
from agentscope.model import ChatModelBase

if TYPE_CHECKING:
    # Only for type hints to avoid runtime circular import
    from games.agent_factory import create_model_from_config

logger = logging.getLogger(__name__)


class SummarizedMemory(MemoryBase):
    """The in-memory memory class with automatic summarization.
    
    When the number of messages exceeds max_messages, this memory will
    automatically call a model to summarize the conversation history,
    clear the old messages, and store the summary as a new message.
    """

    def __init__(
        self,
        **kwargs,
    ) -> None:
        """Initialize the SummarizedMemory object.
        
        Args:
            **kwargs: Keyword arguments. Can contain:
                - memory_config (`Optional[Dict[str, Any]]`): Configuration dictionary with:
                    - system_prompt: System prompt for summarization (optional)
                    - api_key: API key (optional, will use OPENAI_API_KEY env var if not provided)
                    - base_url: API base URL (optional, will use OPENAI_BASE_URL env var if not provided)
                    - model_name: Model name (optional, defaults to 'qwen-plus')
                    - max_messages: Maximum number of messages before summarization (optional, defaults to 40)
                    - temperature: Temperature parameter (optional, defaults to 0.7)
                    - max_tokens: Max tokens parameter (optional, defaults to 2048)
                - Or directly pass these parameters in kwargs (if memory_config is not provided)
        """
        super().__init__()
        self.content: list[Msg] = []
        
        # Load default config from yaml file if exists
        default_config = self._load_default_config()
        
        # Extract memory_config from kwargs if provided
        memory_config = kwargs.get('memory_config')
        
        # Keys that can be overridden via direct kwargs
        override_keys = [
            'system_prompt', 'api_key', 'base_url', 'model_name',
            'max_messages', 'temperature', 'max_tokens', 'summary_prompt',
            'stream',
        ]

        # If memory_config is not provided or is None, try to get config from kwargs directly
        if memory_config is None or not isinstance(memory_config, dict):
            # Extract relevant config from kwargs
            memory_config = {
                k: kwargs[k] for k in override_keys
                if k in kwargs and kwargs[k] is not None
            }
        else:
            # memory_config is provided, allow direct kwargs to override
            direct_overrides = {
                k: kwargs[k] for k in override_keys
                if k in kwargs and kwargs[k] is not None
            }
            memory_config = memory_config.copy()
            memory_config.update(direct_overrides)

        
        # Merge configs with priority: memory_config > default_config (from yaml)
        # Start with default_config, then override with memory_config
        merged_config = default_config.copy()
        merged_config.update(memory_config)
        memory_config = merged_config
        
        # Get configuration with priority: memory_config > environment variables > defaults
        # System prompt for summarization
        self.system_prompt = memory_config.get('system_prompt') or (
            "你是一个专业的对话总结助手，能够准确提取对话中的关键信息。"
        )
        
        # Summary prompt template
        summary_prompt_template = memory_config.get('summary_prompt') or (
            "请总结以下对话历史，保留关键信息和重要细节。"
            "注意：你只需要输出对话的总结内容，不要输出任何 JSON 格式、orders 或其他格式化内容。"
            "只输出纯文本的对话总结。\n\n"
            "对话历史：\n{conversation_history}\n\n"
            "请提供对话总结："
        )
        
        self.summary_prompt = summary_prompt_template
        
        # Max messages
        self.max_messages = memory_config.get('max_messages', 20)
        
        # Build model config from memory_config and environment variables
        full_model_config = {}
        
        # Model name (default: 'qwen-plus')
        full_model_config['model_name'] = memory_config.get('model_name', 'qwen-plus')
        
        # Base URL: memory_config > environment variable
        # Support both 'base_url' and 'url' in memory_config
        base_url = memory_config.get('base_url') or memory_config.get('url')
        if not base_url:
            base_url = os.environ.get('OPENAI_BASE_URL')
        if base_url:
            full_model_config['url'] = base_url
        
        # API key: memory_config > environment variable
        api_key = memory_config.get('api_key')
        if not api_key:
            api_key = os.environ.get('OPENAI_API_KEY')
        if api_key:
            full_model_config['api_key'] = api_key
        
        # Temperature (default: 0.7)
        full_model_config['temperature'] = memory_config.get('temperature', 0.7)
        
        # Max tokens (default: 2048)
        full_model_config['max_tokens'] = memory_config.get('max_tokens', 2048)
        
        # Force stream=False for summarization (we need complete response, not streaming)
        full_model_config['stream'] = memory_config.get('stream', False)
        
        # Create model instance (lazy import to avoid circular dependency)
        self.summary_model: Optional[ChatModelBase] = None
        try:
            from games.agent_factory import create_model_from_config  # noqa: WPS433

            self.summary_model = create_model_from_config(full_model_config)
        except Exception as e:
            logger.warning(
                f"Failed to create summary model: {e}. "
                "Summarization will be disabled."
            )
            self.summary_model = None

    def state_dict(self) -> dict:
        """Convert the current memory into JSON data format."""
        return {
            "content": [_.to_dict() for _ in self.content],
        }

    def load_state_dict(
        self,
        state_dict: dict,
        strict: bool = True,
    ) -> None:
        """Load the memory from JSON data.

        Args:
            state_dict (`dict`):
                The state dictionary to load, which should have a "content"
                field.
            strict (`bool`, defaults to `True`):
                If `True`, raises an error if any key in the module is not
                found in the state_dict. If `False`, skips missing keys.
        """
        self.content = []
        for data in state_dict["content"]:
            data.pop("type", None)
            self.content.append(Msg.from_dict(data))

    async def size(self) -> int:
        """The size of the memory."""
        return len(self.content)

    async def retrieve(self, *args: Any, **kwargs: Any) -> None:
        """Retrieve items from the memory."""
        raise NotImplementedError(
            "The retrieve method is not implemented in "
            f"{self.__class__.__name__} class.",
        )

    async def delete(self, index: Union[Iterable, int]) -> None:
        """Delete the specified item by index(es).

        Args:
            index (`Union[Iterable, int]`):
                The index to delete.
        """
        if isinstance(index, int):
            index = [index]

        invalid_index = [_ for _ in index if 0 > _ or _ >= len(self.content)]

        if invalid_index:
            raise IndexError(
                f"The index {invalid_index} does not exist.",
            )

        self.content = [
            _ for idx, _ in enumerate(self.content) if idx not in index
        ]

    async def add(
        self,
        memories: Union[list[Msg], Msg, None],
        allow_duplicates: bool = False,
    ) -> None:
        """Add message into the memory.

        Args:
            memories (`Union[list[Msg], Msg, None]`):
                The message to add.
            allow_duplicates (`bool`, defaults to `False`):
                If allow adding duplicate messages (with the same id) into
                the memory.
        """
        if memories is None:
            return

        if isinstance(memories, Msg):
            memories = [memories]

        if not isinstance(memories, list):
            raise TypeError(
                f"The memories should be a list of Msg or a single Msg, "
                f"but got {type(memories)}.",
            )

        for msg in memories:
            if not isinstance(msg, Msg):
                raise TypeError(
                    f"The memories should be a list of Msg or a single Msg, "
                    f"but got {type(msg)}.",
                )

        if not allow_duplicates:
            existing_ids = [_.id for _ in self.content]
            memories = [_ for _ in memories if _.id not in existing_ids]
        self.content.extend(memories)

    async def get_memory(self) -> list[Msg]:
        """Get the memory content with automatic summarization.
        
        If the number of messages exceeds max_messages, this method will
        automatically summarize the conversation history, clear old messages,
        and store the summary as a new message.
        
        Returns:
            list[Msg]: The current memory content.
        """
        if len(self.content) > self.max_messages:
            await self._summarize_memory()
        
        return self.content

    async def _summarize_memory(self) -> None:
        """Summarize the conversation history and replace old messages.
        
        The last moderator message (name == "Moderator") is preserved and
        not included in the summary payload.
        """
        original_count = len(self.content)
        summarizable, preserved_tail = self._split_preserve_tail(self.content)

        try:
            summary_text = await self._build_summary_text(summarizable)
            if not summary_text:
                return

            summary_msg = Msg(
                name="system",
                content=f"[对话总结] {summary_text}",
                role="user",
            )

            # Replace old messages with the summary plus any preserved tail
            self.content = [summary_msg, *preserved_tail]

            logger.info(
                "Summarized %s messages into 1 summary message; preserved %s moderator message(s).",
                original_count,
                len(preserved_tail),
            )

        except Exception as e:
            logger.error(
                "Error during memory summarization: %s. Keeping original messages.",
                e,
            )
            import traceback
            logger.debug(traceback.format_exc())
            return

    def _split_preserve_tail(self, messages: list[Msg]) -> tuple[list[Msg], list[Msg]]:
        """Split messages into summarizable part and preserved tail.

        Currently preserves the final message if its name is "Moderator" so
        moderator instructions survive summarization.
        """
        if not messages:
            return [], []

        if getattr(messages[-1], "name", None) == "Moderator":
            return messages[:-1], [messages[-1]]

        return messages, []

    async def _build_summary_text(self, messages: list[Msg]) -> Optional[str]:
        """Generate summary text for provided messages without mutating state."""
        if self.summary_model is None:
            logger.warning(
                "Summary model is not available. Keeping original messages without summarization.",
            )
            return None

        if not messages:
            return None

        conversation_text = self._format_conversation_for_summary(messages)

        prompt_text = self.summary_prompt.format(
            conversation_history=conversation_text,
        )

        payload = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt_text},
        ]

        response = await self.summary_model(payload)
        summary_text = ""

        try:
            import inspect

            if inspect.isasyncgen(response):
                async for chunk in response:
                    if hasattr(chunk, "content"):
                        if isinstance(chunk.content, list):
                            for block in chunk.content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    summary_text += block.get("text", "")
                                elif hasattr(block, "text"):
                                    summary_text += block.text
                        else:
                            summary_text += str(chunk.content)
                    elif hasattr(chunk, "get_text_content"):
                        summary_text += chunk.get_text_content() or ""
                    else:
                        summary_text += str(chunk)
            else:
                raise AttributeError("Not a streaming response")
        except (AttributeError, TypeError, ImportError):
            if hasattr(response, "content"):
                if isinstance(response.content, list):
                    for block in response.content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            summary_text += block.get("text", "")
                        elif hasattr(block, "text"):
                            summary_text += block.text
                    if not summary_text:
                        summary_text = str(response.content)
                else:
                    summary_text = str(response.content)
            elif hasattr(response, "get_text_content"):
                summary_text = response.get_text_content() or ""
            else:
                summary_text = str(response)

        summary_text = summary_text.strip()

        if not summary_text:
            logger.warning("Empty summary generated, keeping original messages")
            return None

        return summary_text

    def _format_conversation_for_summary(self, messages: Optional[list[Msg]] = None) -> str:
        """Format conversation history as text for summarization."""
        msgs = messages if messages is not None else self.content
        lines = []
        for msg in msgs:
            role = getattr(msg, "role", "user")
            content = msg.get_text_content() if hasattr(msg, "get_text_content") else str(msg.content)
            lines.append(f"{role}: {content}")
        
        return "\n".join(lines)

    async def clear(self) -> None:
        """Clear the memory content."""
        self.content = []

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration from yaml file.
        
        Returns:
            Dict[str, Any]: Configuration dictionary loaded from yaml file.
                Returns empty dict if file not found or error occurs.
        """
        # Try to find memory_config.yaml in the same directory as this file
        current_file = Path(__file__)
        config_file = current_file.parent / "memory_config.yaml"
        
        if not config_file.exists():
            # Try to get from environment variable
            env_config_path = os.environ.get('MEMORY_CONFIG_PATH')
            if env_config_path:
                config_file = Path(env_config_path)
        
        if not config_file.exists():
            logger.debug(f"Memory config file not found: {config_file}")
            return {}
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            
            # Extract and flatten config structure
            # Expected structure: {model: {...}, memory: {...}}
            result = {}
            
            # Extract model config
            if 'model' in config and isinstance(config['model'], dict):
                model_cfg = config['model']
                # Support both 'base_url' and 'url' for compatibility
                if 'base_url' in model_cfg:
                    result['base_url'] = model_cfg['base_url']
                elif 'url' in model_cfg:
                    result['base_url'] = model_cfg['url']
                if 'api_key' in model_cfg:
                    result['api_key'] = model_cfg['api_key']
                if 'model_name' in model_cfg:
                    result['model_name'] = model_cfg['model_name']
                if 'temperature' in model_cfg:
                    result['temperature'] = model_cfg['temperature']
                if 'max_tokens' in model_cfg:
                    result['max_tokens'] = model_cfg['max_tokens']
                if 'stream' in model_cfg:
                    result['stream'] = model_cfg['stream']
            
            # Extract memory config
            if 'memory' in config and isinstance(config['memory'], dict):
                memory_cfg = config['memory']
                if 'max_messages' in memory_cfg:
                    result['max_messages'] = memory_cfg['max_messages']
                if 'system_prompt' in memory_cfg:
                    result['system_prompt'] = memory_cfg['system_prompt']
                if 'summary_prompt' in memory_cfg:
                    result['summary_prompt'] = memory_cfg['summary_prompt']
            
            logger.debug(f"Loaded memory config from {config_file}")
            return result
            
        except Exception as e:
            logger.warning(f"Failed to load memory config from {config_file}: {e}")
            return {}
