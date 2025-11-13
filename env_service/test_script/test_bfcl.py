#!/usr/bin/env python3
"""BFCL ground-truth script -- 使用正确格式的 tool_calls。
每一步直接发送函数调用，避免 EnvHandler 再去解析自然语言。"""

from __future__ import annotations
import json
from typing import Dict, List
from env_service.env_client import EnvClient


def tc(name: str, args: Dict, idx: int, turn: int) -> Dict:
    """构造单个 ToolCall 字典"""
    return {
        "id": f"{name}_{turn}_{idx}",
        "name": name,
        "arguments": json.dumps(args, ensure_ascii=False),
        "type": "tool",
        "index": idx,  # ActionMessage 校验所需字段
    }


# --------------------------------------------------------------------------- #
# 每一轮助手回复的 tool_calls 列表
# --------------------------------------------------------------------------- #
ASSISTANT_MESSAGES = [
    # ── Turn-1 ──
    {
        "role": "assistant",
        "content": '<tool_call>\n{"name": "cd", "arguments": {"folder": "document"}}\n</tool_call>\n<tool_call>\n{"name": "mkdir", "arguments": {"dir_name": "temp"}}\n</tool_call>\n<tool_call>\n{"name": "mv", "arguments": {"source": "final_report.pdf", "destination": "temp"}}\n</tool_call>'
    },
    {
        "role": "assistant",
        "content": 'ok.1'
    },
    # ── Turn-2 ──
    {
        "role": "assistant",
        "content": '<tool_call>\n{"name": "cd", "arguments": {"folder": "temp"}}\n</tool_call>\n<tool_call>\n{"name": "grep", "arguments": {"file_name": "final_report.pdf", "pattern": "budget analysis"}}\n</tool_call>'
    },
    {
        "role": "assistant",
        "content": 'ok.2'
    },
    # ── Turn-3 ──
    {
        "role": "assistant",
        "content": '<tool_call>\n{"name": "sort", "arguments": {"file_name": "final_report.pdf"}}\n</tool_call>'
    },
    {
        "role": "assistant",
        "content": 'ok.2'
    },
    # ── Turn-4 ──
    {
        "role": "assistant",
        "content": '<tool_call>\n{"name": "cd", "arguments": {"folder": ".."}}\n</tool_call>\n<tool_call>\n{"name": "mv", "arguments": {"source": "previous_report.pdf", "destination": "temp"}}\n</tool_call>\n<tool_call>\n{"name": "cd", "arguments": {"folder": "temp"}}\n</tool_call>\n<tool_call>\n{"name": "diff", "arguments": {"file_name1": "final_report.pdf", "file_name2": "previous_report.pdf"}}\n</tool_call>'
    },
    {
        "role": "assistant",
        "content": 'ok.2'
    },
]


msg = {
    "role": "assistant",
    "content": (
        "我来帮你查一下。\n<tool_call>\n"
        "{\"name\": \"mkdir\", \"arguments\": {\"dir_name\": \"temp\"}}\n</tool_call>\n<tool_call>\n"
        "{\"name\": \"cd\", \"arguments\": {\"folder\": \"temp\"}}\n</tool_call>"
    )
}
# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    client = EnvClient(base_url="http://localhost:8000")

    # sanity-check 任务是否存在
    env_type = "bfcl"
    task_ids = client.get_env_profile(env_type)
    print(f"Available tasks: {task_ids}")


    # 创建实例
    task_id = "multi_turn_base_0"
    init_response = client.create_instance(env_type, task_id, params={"model_name": "gt-script"})
    print("init state", init_response)
    inst_id = init_response["info"]["instance_id"]
    query = init_response["state"]
    print(f"Created instance {inst_id} with query: {query}")
    # 逐轮交互
    for turn_no, msg in enumerate(ASSISTANT_MESSAGES, 1):
        res = client.step(
            inst_id,
            msg
            # {"role": "assistant", "content": "", "tool_calls": tc_list},
        )
        print(
            f"\n[TURN {turn_no}] term={res['is_terminated']} "
            f"reward={res['reward']}\n state: {res.get('state', {})}"
        )
        if res["is_terminated"]:
            break
        # import pdb;pdb.set_trace()

    # 评估
    score = client.evaluate(inst_id, params={"sparse": True})
    print(f"\n[RESULT] sparse_score = {score}")

    client.release_instance(inst_id)
    print("[DONE] released instance")


if __name__ == "__main__":
    main()
