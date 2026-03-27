"""
best_logger 替代实现 - 原版为 SeeUPO/AgentEvolver 作者内部私有库，未发布到 PyPI。
本模块提供兼容接口，使用 loguru 实现基本功能。
"""
import os
import json
from typing import Any, Dict, List, Optional
from loguru import logger

# 全局配置
_LOGGER_INITIALIZED = False
_BASE_LOG_PATH = "./logs"


def register_logger(
    mods: Optional[List[str]] = None,
    non_console_mods: Optional[List[str]] = None,
    auto_clean_mods: Optional[List[str]] = None,
    base_log_path: str = "./logs",
    debug: bool = False,
    **kwargs
) -> None:
    """初始化 logger 配置。本实现仅记录配置，实际输出到控制台。"""
    global _LOGGER_INITIALIZED, _BASE_LOG_PATH
    if os.environ.get("BEST_LOGGER_INIT"):
        return
    os.environ["BEST_LOGGER_INIT"] = "1"
    _BASE_LOG_PATH = base_log_path
    _LOGGER_INITIALIZED = True
    os.makedirs(base_log_path, exist_ok=True)
    if debug:
        logger.debug(f"best_logger initialized: mods={mods}, base_log_path={base_log_path}")


def _format_dict(data: dict, narrow: bool = False) -> str:
    """格式化 dict 为可读字符串"""
    if narrow:
        return json.dumps(data, ensure_ascii=False, indent=2)
    return json.dumps(data, ensure_ascii=False, indent=2)


def print_dict(
    data: Any,
    header: Optional[str] = None,
    narrow: bool = False,
    mod: Optional[str] = None,
    **kwargs
) -> None:
    """
    打印 dict。支持两种调用方式：
    - print_dict(data, header="xxx", narrow=True, mod="evaluation")
    - print_dict(data, "header")  # 第二个位置参数为 header
    """
    if header is not None and not isinstance(header, str):
        header = str(header)
    formatted = _format_dict(data, narrow) if isinstance(data, dict) else str(data)
    prefix = f"[{header}] " if header else ""
    msg = f"{prefix}{formatted}"
    if mod == "evaluation":
        logger.info(msg)
    else:
        logger.debug(msg)


def _format_listofdict(data_list: List[dict]) -> str:
    """格式化 list of dict 为可读字符串"""
    lines = []
    for i, d in enumerate(data_list):
        role = d.get("role", "?")
        content = d.get("content", d.get("value", str(d)))
        if isinstance(content, str) and len(content) > 200:
            content = content[:200] + "..."
        lines.append(f"  [{i}] {role}: {content}")
    return "\n".join(lines)


def print_listofdict(
    data_list: List[dict],
    mod: Optional[str] = None,
    header: Optional[str] = None,
    **kwargs
) -> None:
    """打印 list of dict（如对话消息列表）"""
    formatted = _format_listofdict(data_list)
    prefix = f"[{header}] " if header else ""
    msg = f"{prefix}\n{formatted}"
    if mod == "exception":
        logger.error(msg)
    elif mod == "c":  # console debug
        logger.debug(msg)
    else:
        logger.info(msg)


# NestedJsonItem 和 SeqItem - 用于 print_nested 的数据结构
class SeqItem:
    """序列项，用于嵌套 JSON 展示"""
    def __init__(self, text=None, title=None, count=None, color=None, **kwargs):
        self.text = text
        self.title = title or text
        self.count = count
        self.color = color
        for k, v in kwargs.items():
            setattr(self, k, v)


class NestedJsonItem:
    """嵌套 JSON 项"""
    def __init__(self, item_id=None, outcome=None, reward=None, content=None, **kwargs):
        self.item_id = item_id
        self.outcome = outcome
        self.reward = reward
        self.content = content
        for k, v in kwargs.items():
            setattr(self, k, v)


def print_nested(
    nested_items: Dict[str, NestedJsonItem],
    main_content: Optional[str] = None,
    header: Optional[str] = None,
    mod: Optional[str] = None,
    narrow: bool = False,
    attach: Optional[str] = None,
    **kwargs
) -> None:
    """打印嵌套结构。本实现简化为 logger 输出。"""
    lines = [f"--- {header or 'Nested'} ---"]
    if main_content:
        lines.append(main_content)
    for key, item in nested_items.items():
        if isinstance(item, NestedJsonItem):
            info = f"  {key}: outcome={getattr(item, 'outcome', '')} reward={getattr(item, 'reward', '')}"
            if item.content and isinstance(item.content, SeqItem):
                text_preview = str(getattr(item.content, 'text', ''))[:100]
                info += f" | {text_preview}..."
            lines.append(info)
        else:
            lines.append(f"  {key}: {item}")
    msg = "\n".join(lines)
    if mod == "evaluation":
        logger.info(msg)
    else:
        logger.debug(msg)
