# -*- coding: utf-8 -*-
"""Echo agent for moderator announcements."""
from typing import Any

from agentscope.agent import AgentBase
from agentscope.message import Msg


class EchoAgent(AgentBase):
    """Echo agent that repeats the input message (Moderator for public announcements)."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "Moderator"

    async def reply(self, content: str) -> Msg:
        """Repeat the input content with its name and role (public moderator announcement)."""
        msg = Msg(
            self.name,
            content,
            role="assistant",
        )
        return msg

    async def handle_interrupt(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Msg:
        """Handle interrupt."""

    async def observe(self, msg: Msg | list[Msg] | None) -> None:
        """Observe the user's message."""

