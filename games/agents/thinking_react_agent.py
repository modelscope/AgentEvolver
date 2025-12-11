"""ReAct agent with private thinking (shared)."""

from typing import Any, Literal
import json
import re

from agentscope.agent import ReActAgent
from agentscope.message import Msg, TextBlock
from agentscope.model import ChatModelBase

from games.agents.utils import extract_text_from_content


class ThinkingReActAgent(ReActAgent):
    """A ReAct agent that thinks before speaking."""

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
        """Initialize a ThinkingReActAgent."""
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

        self._sys_prompt = f"{self._sys_prompt}\n\n{thinking_sys_prompt}"
        self.model_call_history: list[dict[str, Any]] = []

    async def _reasoning(
        self,
        tool_choice: Literal["auto", "none", "any", "required"] | None = None,
    ) -> Msg:
        """Perform reasoning with thinking section."""
        prompt = await self.formatter.format(
            msgs=[
                Msg("system", self.sys_prompt, "system"),
                *await self.memory.get_memory(),
                *await self._reasoning_hint_msgs.get_memory(),
            ],
        )

        msg = await super()._reasoning(tool_choice)

        if msg is not None:
            response_content = extract_text_from_content(msg.content)

            prompt_str = prompt
            if not isinstance(prompt, str):
                if isinstance(prompt, dict):
                    prompt_str = json.dumps(prompt, ensure_ascii=False, indent=2)
                else:
                    prompt_str = str(prompt)

            call_record = {
                "prompt": prompt_str,
                "response": response_content,
                "response_msg": msg.to_dict()
                if hasattr(msg, "to_dict")
                else {
                    "name": msg.name,
                    "content": response_content,
                    "role": msg.role,
                    "timestamp": str(msg.timestamp) if hasattr(msg, "timestamp") else None,
                },
            }
            self.model_call_history.append(call_record)

        if msg is None:
            return msg

        _, public_msg = self._separate_thinking_and_response(msg)

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
        """Separate thinking content from public response."""
        text_content = msg.get_text_content()

        pattern = r"<think>(.*?)</think>"
        matches = re.findall(pattern, text_content, re.DOTALL)

        thinking_content = None
        if matches:
            thinking_content = matches[0].strip()
            public_content = re.sub(pattern, "", text_content, flags=re.DOTALL).strip()
        else:
            public_content = text_content

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

        public_blocks = []
        if isinstance(msg.content, str):
            public_blocks = [
                TextBlock(type="text", text=public_content),
            ]
        elif isinstance(msg.content, list):
            has_non_text = any(block.get("type") != "text" for block in msg.content)

            if has_non_text:
                for block in msg.content:
                    if block.get("type") == "text":
                        block_text = block.get("text", "")
                        if "<think>" in block_text:
                            cleaned_text = re.sub(
                                pattern,
                                "",
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
        public_msg.id = msg.id
        public_msg.timestamp = msg.timestamp

        return thinking_msg, public_msg
