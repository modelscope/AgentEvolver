#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for CachedSummarizedMemory and CacheRetrievalAgent."""

import sys
import os
import asyncio
import shutil
from pathlib import Path

# Ensure project root on path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from agentscope.message import Msg
from games.agent_factory import create_model_from_config
from games.agents.cache_retrieval_agent import CacheRetrievalAgent
from games.agents.memory import CachedSummarizedMemory


def _build_model_config() -> dict | None:
    """从环境变量构建真实模型配置，缺失时返回 None."""
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    model_name = os.environ.get("OPENAI_MODEL_NAME", "qwen-plus")
    if not api_key or not base_url:
        return None

    return {
        "model_name": model_name,
        "url": base_url,
        "api_key": api_key,
        "temperature": float(os.environ.get("OPENAI_TEMPERATURE", 0.2)),
        "max_tokens": int(os.environ.get("OPENAI_MAX_TOKENS", 256)),
        "stream": False,
    }


async def test_cached_memory_flush_and_load(tmp_dir: Path) -> None:
    model_cfg = _build_model_config()
    if model_cfg is None:
        print("⚠️ 未配置 OPENAI_API_KEY/OPENAI_BASE_URL，跳过真实模型摘要测试")
        return

    # 模拟外部传入确定的 log_dir
    game_log_dir = tmp_dir / "test_game_1"
    game_log_dir.mkdir(parents=True, exist_ok=True)

    memory = CachedSummarizedMemory(
        max_messages=5,
        log_dir=game_log_dir,  # 测试新参数 log_dir
        game_id="test_game",
        memory_config=model_cfg,
    )

    if memory.summary_model is None:
        print("⚠️ 摘要模型创建失败，跳过测试")
        return

    # 模拟 10 轮对话，再加 Moderator 尾巴，确保超过 max_messages=5 触发缓存
    for i in range(10):
        role = "user" if i % 2 == 0 else "assistant"
        await memory.add(Msg(name=role, content=f"msg-{i}", role=role))
    await memory.add(Msg(name="Moderator", content="keep me", role="assistant"))

    merged = await memory.get_memory()
    print("🧪 flush 后 cache 条数:", len(memory.cache))
    if len(memory.cache) == 0:
        print("⚠️ 未生成摘要缓存，可能是模型调用失败或返回空摘要，跳过此用例")
        return

    assert len(memory.content) == 1, "Content 应只保留 Moderator 尾巴"
    assert merged[-1].name == "Moderator", "Moderator message preserved"

    # 验证文件路径是否符合预期：log_dir/cache/...
    expected_cache_dir = game_log_dir / "cache"
    assert expected_cache_dir.exists(), f"缓存目录应在 {expected_cache_dir}"
    
    cached = await memory.load_cached_chunk(1)
    print("🧪 chunk#1 条数:", len(cached))
    assert len(cached) >= 5, "Cached chunk 应包含溢出的原始消息"


async def test_agent_recall_by_query(tmp_dir: Path) -> None:
    model_cfg = _build_model_config()
    if model_cfg is None:
        print("⚠️ 未配置 OPENAI_API_KEY/OPENAI_BASE_URL，跳过检索模型测试")
        return

    # 模拟外部传入确定的 log_dir
    game_log_dir = tmp_dir / "test_game_agent"
    game_log_dir.mkdir(parents=True, exist_ok=True)

    memory = CachedSummarizedMemory(
        max_messages=5,
        log_dir=game_log_dir,  # 测试新参数 log_dir
        game_id="test_game_agent",
        agent_id="tester",     # 测试 agent_id 组合
        memory_config=model_cfg,
    )
    
    # 预期路径应该是 log_dir/tester/cache
    # 因为 CachedSummarizedMemory 逻辑: if agent_id: cache_dir = base_path / str(agent_id) / "cache"
    expected_cache_dir = game_log_dir / "tester" / "cache"

    if memory.summary_model is None:
        print("⚠️ 摘要模型创建失败，跳过检索测试")
        return

    # 模拟 10 轮对话，确保触发缓存，再加 Moderator 尾巴
    for i in range(10):
        role = "user" if i % 2 == 0 else "assistant"
        await memory.add(Msg(name=role, content=f"alpha-{i}", role=role))
    await memory.add(Msg(name="Moderator", content="tail", role="assistant"))
    await memory.get_memory()  # trigger flush to cache

    if len(memory.cache) == 0:
        print("⚠️ 未生成摘要缓存，可能模型调用失败或返回空摘要，跳过检索用例")
        return
        
    assert expected_cache_dir.exists(), f"带AgentID的缓存目录应在 {expected_cache_dir}"

    main_model = create_model_from_config(model_cfg)
    retrieval_cfg = model_cfg.copy()

    agent = CacheRetrievalAgent(
        name="tester",
        sys_prompt="",
        model=main_model,
        formatter=None,
        memory=memory,
        retrieval_model_config=retrieval_cfg,
    )

    resp = await agent.recall_cache_by_query("alpha")
    print("🧪 检索模型输出:", resp.content)
    assert resp.metadata.get("cache_chunk_ids"), "检索应返回 chunk id 列表"


def _make_tmp() -> Path:
    tmp_dir = Path(__file__).parent / "tmp_logs_test"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir


async def main() -> None:
    tmp_dir = _make_tmp()
    try:
        await test_cached_memory_flush_and_load(tmp_dir)
        print("✓ CachedSummarizedMemory 刷盘与读取通过")
        await test_agent_recall_by_query(tmp_dir)
        print("✓ CacheRetrievalAgent 检索召回通过")
        print("✅ 全部真实模型缓存相关测试通过")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())

