# -*- coding: utf-8 -*-
"""Agent that can recall cached conversation chunks from CachedSummarizedMemory."""

import logging
from typing import Any, Literal, Optional, TYPE_CHECKING

from agentscope.message import Msg
from agentscope.model import ChatModelBase

from games.agents.memory.CachedSummarizedMemory import CachedSummarizedMemory
from games.agents.thinking_react_agent import ThinkingReActAgent
from games.agents.utils import extract_text_from_content

if TYPE_CHECKING:
    # Only for type hints to avoid runtime circular import
    from games.agent_factory import create_model_from_config

logger = logging.getLogger(__name__)


class CacheRetrievalAgent(ThinkingReActAgent):
    """ThinkingReActAgent with optional cache recall capability.

    - If memory is a CachedSummarizedMemory, the agent can load cached chunks.
    - Otherwise it behaves exactly like ThinkingReActAgent.
    """

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
        thinking_tag_start_end: tuple[str, str] = ("<think>", "</think>"),
        retrieval_model_config: Optional[dict[str, Any]] = None,
        retrieval_prompt: Optional[str] = None,
        retrieval_top_k: int = 1,
        retrieval_model: Optional[ChatModelBase] = None,
    ) -> None:
        if memory is None:
            memory = CachedSummarizedMemory()

        self._cache_enabled = isinstance(memory, CachedSummarizedMemory)
        self.retrieval_top_k = retrieval_top_k
        self.retrieval_prompt = retrieval_prompt or (
            "给定查询和已缓存对话摘要，请返回最相关的缓存chunk_id列表（用逗号分隔）。"
            "只输出数字id，按相关性降序，最多 {top_k} 个。"
        )
        # Prefer injected retrieval_model; fallback to config with lazy import to avoid circulars
        self.retrieval_model = retrieval_model
        if self.retrieval_model is None and retrieval_model_config:
            try:
                from games.agent_factory import create_model_from_config  # noqa: WPS433

                self.retrieval_model = create_model_from_config(retrieval_model_config)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to create retrieval model: %s", exc)
                self.retrieval_model = None

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
            thinking_sys_prompt=thinking_sys_prompt,
            thinking_tag_start_end=thinking_tag_start_end,
        )

    async def recall_cache_chunk(self, chunk_id: int) -> Msg:
        """Load a cached chunk by id when cache is enabled; otherwise warn."""
        if not self._cache_enabled:
            fallback = Msg(
                name=self.name,
                content="[cache] Cache not available with current memory backend.",
                role="assistant",
            )
            await self.memory.add(fallback)
            return fallback

        cached_messages = await self.memory.load_cached_chunk(chunk_id)
        if not cached_messages:
            response = Msg(
                name=self.name,
                content=f"[cache] No cached chunk found for id {chunk_id}.",
                role="assistant",
            )
            await self.memory.add(response)
            return response

        formatted_text = self._format_cached_messages(cached_messages)
        response = Msg(
            name=self.name,
            content=formatted_text,
            role="assistant",
            metadata={"cache_chunk_id": chunk_id},
        )
        await self.memory.add(response)
        return response

    def _format_cached_messages(self, messages: list[Msg]) -> str:
        lines: list[str] = []
        for msg in messages:
            role = getattr(msg, "role", "user")
            name = getattr(msg, "name", role)
            content = extract_text_from_content(msg.content)
            lines.append(f"{name} ({role}): {content}")
        return "\n".join(lines)

    def cache_overview(self) -> list[dict[str, Any]]:
        """Expose cached chunk metadata to callers (empty if unsupported)."""
        if not self._cache_enabled:
            return []
        return self.memory.get_cache_overview()

    async def recall_cache_by_query(self, query: str, top_k: Optional[int] = None) -> Msg:
        """Use retrieval model to pick which cached chunks to load based on query.

        - If cache未启用或未配置检索模型，会给出提示。
        - 选择到的 chunk 会加载原文并返回给 agent。
        """
        if not self._cache_enabled:
            return await self._emit_cache_disabled()

        if self.retrieval_model is None:
            return await self._emit_no_retrieval_model()

        overview = self.memory.get_cache_overview()
        if not overview:
            return await self._emit_simple_msg("[cache] No cached chunks available for retrieval.")

        k = top_k or self.retrieval_top_k or 1
        chosen_ids = await self._select_cache_ids(query, overview, k)
        if not chosen_ids:
            return await self._emit_simple_msg("[cache] Retrieval model returned no chunk ids.")

        loaded_msgs: list[Msg] = []
        for cid in chosen_ids:
            loaded_msgs.extend(await self.memory.load_cached_chunk(cid))

        if not loaded_msgs:
            return await self._emit_simple_msg(
                f"[cache] Selected ids {chosen_ids}, but no cached content found."
            )

        formatted_text = self._format_cached_messages(loaded_msgs)
        response = Msg(
            name=self.name,
            content=formatted_text,
            role="assistant",
            metadata={"cache_chunk_ids": chosen_ids},
        )
        await self.memory.add(response)
        return response

    async def _select_cache_ids(self, query: str, overview: list[dict[str, Any]], top_k: int) -> list[int]:
        """Call retrieval model to choose chunk ids based on query and overview."""
        lines = []
        for item in overview:
            cid = item.get("chunk_id")
            preview = item.get("summary_preview", "")
            count = item.get("message_count", "?")
            lines.append(f"chunk_id={cid}, messages={count}, summary_preview={preview}")
        overview_text = "\n".join(lines)

        user_prompt = (
            f"查询: {query}\n"
            f"可选缓存摘要:\n{overview_text}\n"
            f"请选出最相关的 chunk_id，最多 {top_k} 个，用逗号分隔只输出数字。"
        )

        prompt = self.retrieval_prompt.format(top_k=top_k) if "{top_k}" in self.retrieval_prompt else self.retrieval_prompt

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_prompt},
        ]

        result_text = await self._call_retrieval_model(messages)
        print("检索模型返回原始文本:", result_text)
        return self._parse_chunk_ids(result_text, overview)

    async def _call_retrieval_model(self, messages: list[dict[str, Any]]) -> str:
        """Run retrieval model; supports streaming/non-streaming like summary path."""
        response = await self.retrieval_model(messages)
        text = ""
        try:
            import inspect
            if inspect.isasyncgen(response):
                async for chunk in response:
                    if hasattr(chunk, "content"):
                        if isinstance(chunk.content, list):
                            for block in chunk.content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    text += block.get("text", "")
                                elif hasattr(block, "text"):
                                    text += block.text
                        else:
                            text += str(chunk.content)
                    elif hasattr(chunk, "get_text_content"):
                        text += chunk.get_text_content() or ""
                    else:
                        text += str(chunk)
            else:
                raise AttributeError("Not a streaming response")
        except (AttributeError, TypeError, ImportError):
            if hasattr(response, "content"):
                if isinstance(response.content, list):
                    for block in response.content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text += block.get("text", "")
                        elif hasattr(block, "text"):
                            text += block.text
                    if not text:
                        text = str(response.content)
                else:
                    text = str(response.content)
            elif hasattr(response, "get_text_content"):
                text = response.get_text_content() or ""
            else:
                text = str(response)

        return text.strip()

    def _parse_chunk_ids(self, raw: str, overview: list[dict[str, Any]]) -> list[int]:
        """Parse comma/space-separated ids and filter to existing overview ids."""
        if not raw:
            return []
        candidates = []
        for token in raw.replace("\n", " ").replace(",", " ").split():
            try:
                candidates.append(int(token))
            except ValueError:
                continue
        valid_ids = {item.get("chunk_id") for item in overview if "chunk_id" in item}
        return [cid for cid in candidates if cid in valid_ids][: max(len(candidates), 1)]

    async def _emit_cache_disabled(self) -> Msg:
        msg = Msg(
            name=self.name,
            content="[cache] Cache not available with current memory backend.",
            role="assistant",
        )
        await self.memory.add(msg)
        return msg

    async def _emit_no_retrieval_model(self) -> Msg:
        msg = Msg(
            name=self.name,
            content="[cache] Retrieval model not configured; cannot select cached chunks.",
            role="assistant",
        )
        await self.memory.add(msg)
        return msg

    async def _emit_simple_msg(self, content: str) -> Msg:
        msg = Msg(name=self.name, content=content, role="assistant")
        await self.memory.add(msg)
        return msg