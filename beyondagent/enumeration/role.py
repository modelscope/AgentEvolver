from enum import Enum


class Role(str, Enum):
    """
    Enumeration for different roles in the system.

    Attributes:
        SYSTEM (str): Represents the system role.
        USER (str): Represents the user role.
        TOOL (str): Represents the tool role.
        ASSISTANT (str): Represents the assistant role.
        CONTEXT_ASSISTANT (str): Represents the context assistant role.
        SUMMARY_ASSISTANT (str): Represents the summary assistant role.
    """
    SYSTEM = "system"  # ⭐ Defines the system role
    USER = "user"  # ⭐ Defines the user role
    TOOL = "tool"  # ⭐ Defines the tool role

    ASSISTANT = "assistant"  # ⭐ Defines the assistant role
    CONTEXT_ASSISTANT = "context_assistant"  # ⭐ Defines the context assistant role
    SUMMARY_ASSISTANT = "summary_assistant"  # ⭐ Defines the summary assistant role
