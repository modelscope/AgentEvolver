from enum import Enum


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    TOOL = "tool"

    ASSISTANT = "assistant"
    CONTEXT_ASSISTANT = "context_assistant"
    SUMMARY_ASSISTANT = "summary_assistant"
