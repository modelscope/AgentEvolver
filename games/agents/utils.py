"""Utilities shared by agent implementations."""

from typing import Any


def extract_text_from_content(content: str | list[Any] | None) -> str:
    """Convert mixed message content into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content", "")))
            else:
                parts.append(str(item))
        return " ".join(parts)

    return str(content)


__all__ = ["extract_text_from_content"]
