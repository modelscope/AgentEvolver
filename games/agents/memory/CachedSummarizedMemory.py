# -*- coding: utf-8 -*-
"""Memory that caches overflowed conversations to disk with summaries."""

import json
import logging
import time
from pathlib import Path
from typing import Any

from agentscope.message import Msg

from games.agents.memory.SummarizedMemory import SummarizedMemory

logger = logging.getLogger(__name__)


class CachedSummarizedMemory(SummarizedMemory):
    """Summarized memory that also persists overflowed messages to disk.

    When content grows beyond `max_messages`, the current content is saved to
    `logs/<game_id>/cache/chunk_xxxx/` as JSON (full content) and text (summary),
    a summary message is stored in `self.cache`, and `self.content` is trimmed
    (preserving the Moderator tail if present).
    """

    def __init__(self, **kwargs: Any) -> None:
        self.cache: list[Msg] = []
        self.cache_metadata: list[dict[str, Any]] = []
        self.game_id = kwargs.get("game_id") or "default_game"
        # Optional agent-level identifier to further separate caches.
        # If provided (e.g., per-agent name or id), cached files will be
        # stored under logs/<game_id>/<agent_id>/cache/ instead of
        # logs/<game_id>/cache/ to avoid cross-agent overwrites.
        self.agent_id = kwargs.get("agent_id")
        
        # If log_dir is provided directly, use it as the base path
        # This aligns with the game's logging directory structure
        log_dir = kwargs.get("log_dir")
        
        if log_dir:
            # If log_dir is provided, cache_dir is directly inside it
            # We still append agent_id to avoid conflicts in shared log_dir
            base_path = Path(log_dir)
        else:
            # Fallback to constructing path from logs_root + game_id
            logs_root = kwargs.get("logs_root")
            base_root = Path(logs_root) if logs_root else Path(__file__).parent / "logs"
            base_path = base_root / str(self.game_id)

        if self.agent_id:
            self.cache_dir = base_path / str(self.agent_id) / "cache"
        else:
            self.cache_dir = base_path / "cache"
            
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._chunk_counter = self._init_chunk_counter()

        super().__init__(**kwargs)

    async def get_memory(self) -> list[Msg]:
        """Return cache summaries + live content, flushing to cache if needed."""
        if len(self.content) > self.max_messages:
            await self._flush_to_cache()
        return [*self.cache, *self.content]

    async def clear(self) -> None:
        """Clear in-memory content and cached summaries (disk files remain)."""
        await super().clear()
        self.cache = []
        self.cache_metadata = []
        self._chunk_counter = self._init_chunk_counter()

    async def load_cached_chunk(self, chunk_id: int) -> list[Msg]:
        """Load original messages from a cached chunk on disk."""
        chunk_dir = self._chunk_dir(chunk_id)
        content_path = chunk_dir / "content.json"
        if not content_path.exists():
            return []

        try:
            with open(content_path, "r", encoding="utf-8") as f:
                raw = json.load(f) or []
        except Exception as exc:  # pragma: no cover - defensive read path
            logger.error("Failed to read cached chunk %s: %s", chunk_id, exc)
            return []

        loaded: list[Msg] = []
        for item in raw:
            if isinstance(item, dict):
                item.pop("type", None)
                try:
                    loaded.append(Msg.from_dict(item))
                    continue
                except Exception:
                    pass
                loaded.append(
                    Msg(
                        name=item.get("name", "system"),
                        content=item.get("content", ""),
                        role=item.get("role", "user"),
                    )
                )
            else:
                loaded.append(Msg(name="system", content=str(item), role="user"))
        return loaded

    def get_cache_overview(self) -> list[dict[str, Any]]:
        """Return metadata for cached chunks (ids, paths, counts)."""
        return list(self.cache_metadata)

    async def _flush_to_cache(self) -> None:
        """Persist current content to disk and store summary in cache list."""
        summarizable, preserved_tail = self._split_preserve_tail(self.content)
        if not summarizable:
            self.content = preserved_tail
            return

        summary_text = await self._build_summary_text(summarizable)
        if not summary_text:
            return

        chunk_id = self._next_chunk_id()
        chunk_dir = self._chunk_dir(chunk_id)
        chunk_dir.mkdir(parents=True, exist_ok=True)
        content_path = chunk_dir / "content.json"
        summary_path = chunk_dir / "summary.txt"

        serialized = self._serialize_messages(summarizable)
        with open(content_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, ensure_ascii=False, indent=2)
        summary_path.write_text(summary_text, encoding="utf-8")

        summary_msg = Msg(
            name="system",
            content=f"[缓存对话总结#{chunk_id}] {summary_text}",
            role="user",
            metadata={
                "cache_chunk_id": chunk_id,
                "content_path": str(content_path),
                "summary_path": str(summary_path),
                "created_at": time.time(),
            },
        )

        self.cache.append(summary_msg)
        self.cache_metadata.append(
            {
                "chunk_id": chunk_id,
                "content_path": str(content_path),
                "summary_path": str(summary_path),
                "message_count": len(summarizable),
                "summary_preview": summary_text[:80],
            }
        )

        self.content = preserved_tail
        logger.info(
            "Cached %s messages to %s and kept %s moderator message(s) in memory.",
            len(summarizable),
            content_path,
            len(preserved_tail),
        )

    def state_dict(self) -> dict:
        """Include cache summaries and metadata in serialization."""
        data = super().state_dict()
        data["cache"] = self._serialize_messages(self.cache)
        data["cache_metadata"] = self.cache_metadata
        return data

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        """Restore memory content and cache summaries from state."""
        super().load_state_dict(state_dict, strict)
        self.cache = []
        for data in state_dict.get("cache", []):
            if isinstance(data, dict):
                data.pop("type", None)
                try:
                    self.cache.append(Msg.from_dict(data))
                    continue
                except Exception:
                    pass
                self.cache.append(
                    Msg(
                        name=data.get("name", "system"),
                        content=data.get("content", ""),
                        role=data.get("role", "user"),
                    )
                )
        self.cache_metadata = state_dict.get("cache_metadata", [])
        self._chunk_counter = self._init_chunk_counter()

    def _init_chunk_counter(self) -> int:
        """Determine next chunk id from existing cache folders."""
        existing = [p for p in self.cache_dir.glob("chunk_*") if p.is_dir()]
        max_id = 0
        for path in existing:
            try:
                suffix = path.name.split("_")[1]
                max_id = max(max_id, int(suffix))
            except (IndexError, ValueError):
                continue
        return max_id

    def _next_chunk_id(self) -> int:
        self._chunk_counter += 1
        return self._chunk_counter

    def _chunk_dir(self, chunk_id: int) -> Path:
        return self.cache_dir / f"chunk_{chunk_id:04d}"

    def _serialize_messages(self, messages: list[Msg]) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for msg in messages:
            if hasattr(msg, "to_dict"):
                data = msg.to_dict()
            else:
                data = {"name": getattr(msg, "name", "system"), "content": getattr(msg, "content", ""), "role": getattr(msg, "role", "user")}
            data.pop("type", None)
            payload.append(data)
        return payload