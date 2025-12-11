"""Terminal User Agent that prints all observed messages (shared)."""

from typing import Optional, Type

from pydantic import BaseModel

from agentscope.agent import UserAgent
from agentscope.message import Msg
from agentscope.memory import InMemoryMemory, MemoryBase

from games.agents.utils import extract_text_from_content


class TerminalUserAgent(UserAgent):
    """User Agent that prints all observed messages to terminal."""

    def __init__(
        self,
        name: str,
        memory: Optional[MemoryBase] = None,
    ) -> None:
        """Initialize TerminalUserAgent."""
        super().__init__(name=name)
        self.memory = memory if memory is not None else InMemoryMemory()

    async def reply(
        self,
        msg: Msg | list[Msg] | None = None,
        structured_model: Type[BaseModel] | None = None,
    ) -> Msg:
        """Receive input message(s) and generate a reply message from the user."""
        if msg is not None:
            if isinstance(msg, Msg):
                await self.memory.add(msg)
            elif isinstance(msg, list):
                for m in msg:
                    if isinstance(m, Msg):
                        await self.memory.add(m)

        input_data = await self._input_method(
            agent_id=self.id,
            agent_name=self.name,
            structured_model=structured_model,
        )

        blocks_input = input_data.blocks_input
        if (
            blocks_input
            and len(blocks_input) == 1
            and blocks_input[0].get("type") == "text"
        ):
            blocks_input = blocks_input[0].get("text")

        input_content = extract_text_from_content(blocks_input)
        print("-" * 70)
        print(input_content)
        print("-" * 70)

        reply_msg = Msg(
            self.name,
            content=blocks_input,
            role="user",
            metadata=input_data.structured_input,
        )

        await self.memory.add(reply_msg)

        return reply_msg

    async def observe(self, msg: Msg | list[Msg] | None) -> None:
        """Observe messages and print them to terminal."""
        if msg is None:
            return

        if isinstance(msg, Msg):
            await self.memory.add(msg)
        elif isinstance(msg, list):
            for m in msg:
                if isinstance(m, Msg):
                    await self.memory.add(m)

        if isinstance(msg, Msg):
            content = extract_text_from_content(msg.content)
            print("-" * 70)
            print(f"{msg.name}: {content}")
            print("-" * 70)
        elif isinstance(msg, list):
            if not msg:
                return
            print("-" * 70)
            for m in msg:
                if isinstance(m, Msg):
                    content = extract_text_from_content(m.content)
                    print(f"{m.name}: {content}")
            print("-" * 70)

        await super().observe(msg)
