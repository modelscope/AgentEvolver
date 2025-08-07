import torch
import verl.utils.torch_functional as verl_F
from openai import AsyncOpenAI
import os
import json
from pathlib import Path
from loguru import logger
import time
import traceback
from tqdm import tqdm
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Optional, Literal
import threading
from dataclasses import dataclass, asdict
import random

__all__ = [
    "evaluate_step_flags_parallel",
    "apply_step_mask_vectorized", 
    "ParallelSemanticProcessor",
]

@dataclass
class EvaluationTask:
    """评估任务的数据结构 - 一次评估整个sample的所有steps"""
    sample_idx: int
    query: str
    rollout: str
    steps: List[str]  # 所有step的文本列表
    overall_adv: float

@dataclass
class EvaluationResult:
    """评估结果的数据结构 - 包含整个sample的所有step结果"""
    sample_idx: int
    step_results: List[bool]  # 所有steps的评估结果
    response_time: float

@dataclass
class EvaluationRecord:
    """评估记录的数据结构，用于保存到文件"""
    sample_idx: int
    query: str
    rollout: str
    steps: List[str]
    overall_adv: float
    llm_input_messages: List[Dict]
    llm_raw_output: str
    llm_parsed_results: List[bool]  # 所有steps的解析结果
    response_time: float
    timestamp: float
    model_name: str
    evaluation_type: str
    global_step: Optional[int] = None
    epoch: Optional[str] = None

# =========================================================
# Added: rollout parsing & batch-eval prompt utilities
# =========================================================
import re 

def parse_rollout_to_steps(rollout: str) -> List[Dict[str, str]]:
    """
    将包含 ... assistant\n<action text>\nuser\n<observation text> ... 的长串 rollout 拆成步骤列表。
    """
    parts = re.split(r'\n(assistant|user)\n', rollout, flags=re.I)

    if parts and parts[0].strip():
        parts = ['assistant', parts[0]] + parts[1:]

    steps: List[Dict[str, str]] = []
    i = 0
    while i < len(parts) - 1:
        role, text = parts[i].lower(), parts[i + 1]
        if role == 'assistant':
            action = text.strip()
            observation = ''
            if i + 2 < len(parts) - 1 and parts[i + 2].lower() == 'user':
                observation = parts[i + 3].strip()
                i += 2
            steps.append({"action": action, "observation": observation})
        i += 2
    return steps

def build_batch_evaluation_prompt(
        query: str,
        steps: List[Dict[str, str]],
        overall_adv: float,
        max_step_chars: int = 2000,
) -> List[dict]:
    """
    • SYSTEM message: 说明输入字段与评估任务
    • USER  message : 实际传入的数据
    """
    polarity = "positive" if overall_adv > 0 else "negative"

    # ---------------- system ----------------
    sys_msg = (
        "You are an expert *process* reward evaluator.\n\n"
        "The single message you receive always contains three labelled sections:\n"
        "  1. OVERALL ADVANTAGE – a scalar summarising the final answer quality.\n"
        "  2. TASK DESCRIPTION   – the user’s original request.\n"
        "  3. SOLUTION TRAJECTORY – a numbered list of assistant steps.\n\n"
        "Evaluation rule:\n"
        "• If OVERALL ADVANTAGE is **positive (> 0)**, judge each step by whether its ACTION\n"
        "  makes the overall answer *even better* than before (incremental improvement).\n"
        "• If OVERALL ADVANTAGE is **negative (< 0)**, judge each step by whether it *actively\n"
        "  corrects the existing error*. Mark GOOD **only** when the ACTION clearly fixes or\n"
        "  moves the answer towards correctness; otherwise mark BAD.\n\n"
        "Ignore superficial politeness or formatting. Focus strictly on the technical impact\n"
        "of the ACTION (and OBSERVATION if present).\n\n"
        "Reply IN THE REQUIRED OUTPUT FORMAT and output nothing else."
    )

    # ---------------- helpers ----------------
    def _trim(txt: str) -> str:
        return txt if len(txt) <= max_step_chars else txt[:max_step_chars] + "\n…"

    # ---------------- user ----------------
    user_parts = [
        f"**OVERALL ADVANTAGE {overall_adv:+.4f} ({polarity})**",
        "",
        "### TASK DESCRIPTION",
        query,
        "",
        f"### SOLUTION TRAJECTORY  (total {len(steps)} steps)",
    ]

    for idx, st in enumerate(steps):
        block = [
            f">>> EVAL-STEP {idx} <<<",
            "<|ACTION|>",
            _trim(st['action']),
            "<|END|>",
        ]
        if st['observation']:
            block += [
                "<|OBSERVATION|>",
                _trim(st['observation']),
                "<|END|>",
            ]
        user_parts.append("\n".join(block))

    user_parts += [
        "",
        "---",
        "Evaluation reminder:",
        "• Positive overall advantage → ask: *Does this step further improve the answer?*",
        "• Negative overall advantage → ask: *Does this step clearly correct the error?*",
        "",
        "REQUIRED OUTPUT FORMAT:",
        "Step 0 Analysis: <your reasoning>",
        "Step 0 Judgment: GOOD/BAD",
        "",
        "Step 1 Analysis: <your reasoning>",
        "Step 1 Judgment: GOOD/BAD",
        "",
        "[…continue for all steps…]",
    ]

    return [
        {"role": "system", "content": sys_msg},
        {"role": "user",   "content": "\n".join(user_parts)},
    ]


def build_batch_evaluation_prompt_from_rollout(query: str, rollout: str, overall_adv: float, max_step_chars: int = 2000):
    steps = parse_rollout_to_steps(rollout)
    return build_batch_evaluation_prompt(query, steps, overall_adv, max_step_chars)

def parse_batch_evaluation_result(response: str, num_steps: int):
    numbered = {}
    for m in re.finditer(r"Step\s+(\d+)\s+Judgment:\s*(GOOD|BAD)", response, flags=re.I):
        numbered[int(m.group(1))] = m.group(2).upper() == "GOOD"
    if len(numbered) == num_steps:
        return [numbered[i] for i in range(num_steps)]
    flags = re.findall(r"\b(GOOD|BAD)\b", response.upper())
    if len(flags) >= num_steps:
        return [flag == "GOOD" for flag in flags[:num_steps]]
    raise ValueError("Could not parse evaluation result")

def _get_overall_advantage(advantages_tensor, mask=None):
    """从advantages tensor中获取overall advantage值"""
    if advantages_tensor.dim() == 0:
        return advantages_tensor.item()
    
    if advantages_tensor.dim() == 1:
        if mask is not None:
            valid_advantages = advantages_tensor[mask.bool()]
            if len(valid_advantages) > 0:
                return valid_advantages[0].item()
            else:
                return 0.0
        else:
            non_zero_mask = torch.abs(advantages_tensor) > 1e-8
            if non_zero_mask.any():
                return advantages_tensor[non_zero_mask][0].item()
            else:
                return 0.0
    
    raise ValueError(f"Unsupported advantages_tensor shape: {advantages_tensor.shape}")

def _save_evaluation_record(record: EvaluationRecord, save_dir: Optional[str] = None):
    """保存评估记录到文件"""
    if save_dir is None:
        return
    
    try:
        base_save_path = Path(save_dir)
        base_save_path.mkdir(parents=True, exist_ok=True)
        
        if record.global_step is not None:
            step_subdir = f"step_{record.global_step:06d}"
        else:
            step_subdir = "step_unknown"
        
        step_save_path = base_save_path / step_subdir
        step_save_path.mkdir(parents=True, exist_ok=True)
        
        timestamp_str = f"{record.timestamp:.3f}".replace('.', '_')
        global_step_str = f"step{record.global_step:06d}" if record.global_step is not None else "nostep"
        filename = f"{global_step_str}_sample{record.sample_idx:03d}_{timestamp_str}.json"
        
        file_path = step_save_path / filename
        record_dict = asdict(record)
        record_dict["_metadata"] = {
            "save_time": time.time(),
            "step_directory": step_subdir,
            "file_name": filename,
            "full_path": str(file_path),
            "num_steps": len(record.steps)
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(record_dict, f, ensure_ascii=False, indent=2)
        
        print(f"[record_save] ✅ Saved sample {record.sample_idx} with {len(record.steps)} steps: {step_subdir}/{filename}")
            
    except Exception as e:
        print(f"[record_save] ❌ FAILED to save evaluation record for sample {record.sample_idx}: {e}")
        print(f"[record_save] 📁 Path: {save_dir}")

# ————————————————————————————————————————————————————————————————
# API评估 - 增强的重试机制
# ————————————————————————————————————————————————————————————————

async def _async_safe_query(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    semaphore: asyncio.Semaphore,
    max_retries: int = 200,
    timeout_s: int = 120,         # 可调：单次请求超时阈值
) -> str:
    """异步 API 调用：对 **所有** 异常重试，429 用指数退避"""
    async with semaphore:
        last_exception = None

        for attempt in range(max_retries):
            try:
                # ---------- 普通 / thinking 模型分支 ----------
                is_thinking_model = model.lower() in {
                    "qwq-plus",
                    "qwen3-30b-a3b-thinking-2507",
                    "qwen3-235b-a22b-thinking-2507",
                }

                if is_thinking_model:
                    print(f"[API] Using streaming mode for thinking model: {model}")
                    response = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.0,
                        extra_body={"enable_thinking": True},
                        stream=True,
                        max_tokens=10000,
                        timeout=timeout_s,
                    )

                    answer_content, reasoning_content = "", ""
                    async for chunk in response:
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        if getattr(delta, "reasoning_content", None):
                            reasoning_content += delta.reasoning_content
                        if getattr(delta, "content", ""):
                            answer_content += delta.content

                    final_answer = answer_content.strip()
                    return final_answer

                else:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.0,
                        timeout=timeout_s,
                        max_tokens=10000,
                    )
                    return response.choices[0].message.content.strip()

            # ---------- 统一异常处理 ----------
            except Exception as e:                # ❶ 捕获所有异常
                last_exception = e
                err = str(e).lower()

                is_rate_limit = any(
                    key in err
                    for key in [
                        "429",
                        "rate limit",
                        "exceeded your current requests",
                        "limit_requests",
                    ]
                )

                # ----------- 若未到最大重试次数 -----------
                if attempt < max_retries - 1:
                    # 429 → 指数退避 + 抖动
                    if is_rate_limit:
                        backoff = min(1.5 ** attempt, 60)       # 上限 60 s
                        jitter  = backoff * 0.25 * random.random()
                        wait    = backoff + jitter
                        print(f"[API Retry] 429 (attempt {attempt+1}/{max_retries}) "
                              f"sleep {wait:.1f}s")
                        await asyncio.sleep(wait)
                    else:
                        # 其它异常 → 线性退避
                        wait = min(2.0 * (attempt + 1), 15)
                        print(f"[API Retry] {type(e).__name__}: {e} "
                              f"(attempt {attempt+1}/{max_retries}) sleep {wait:.1f}s")
                        await asyncio.sleep(wait)
                    # 继续 for-loop
                else:
                    # ----------- 已达到最大重试次数 -----------
                    print(f"[API Error] ❌ Max retries ({max_retries}) exceeded: {e}")
                    break

        # loop 结束仍失败 → 抛出最后一次异常
        raise last_exception


async def _evaluate_single_sample_api(
    client: AsyncOpenAI,
    model_name: str,
    task: EvaluationTask,
    semaphore: asyncio.Semaphore,
    max_retries: int = 200,
    save_dir: Optional[str] = None,
    global_step: Optional[int] = None,
    epoch: Optional[str] = None
) -> EvaluationResult:
    """使用 API 一次性评估单个 sample 的所有 steps"""
    start_time = time.time()

    try:
        # 1) 构造批量评估 prompt
        messages = build_batch_evaluation_prompt_from_rollout(
            task.query, task.rollout, task.overall_adv
        )
        # 2) 调用 LLM
        llm_raw_output = await _async_safe_query(
            client, model_name, messages, semaphore, max_retries
        )

        # 3) 解析结果
        try:
            step_results = parse_batch_evaluation_result(
                llm_raw_output, len(task.steps)
            )
            print(
                f"[API] ✅ Sample {task.sample_idx}: Successfully parsed "
                f"{len(step_results)} step results"
            )
        except Exception as parse_error:
            # ——> 解析失败：不做任何缩放（全部使用 “无缩放” 标记）
            print(
                f"[API] ❌ Sample {task.sample_idx}: Parse error, "
                f"disable rescale: {parse_error}"
            )
            uniform_flag = task.overall_adv > 0  # True=GOOD, False=BAD
            step_results = [uniform_flag for _ in task.steps]

        response_time = time.time() - start_time

        # 4) 保存评估记录（保持不变）
        if save_dir:
            is_thinking_model = model_name.lower() in {
                "qwq-plus",
                "qwen3-30b-a3b-thinking-2507",
                "qwen3-235b-a22b-thinking-2507",
            }
            record = EvaluationRecord(
                sample_idx=task.sample_idx,
                query=task.query,
                rollout=task.rollout,
                steps=task.steps,
                overall_adv=task.overall_adv,
                llm_input_messages=messages,
                llm_raw_output=llm_raw_output,
                llm_parsed_results=step_results,
                response_time=response_time,
                timestamp=time.time(),
                model_name=f"{model_name}{'_thinking' if is_thinking_model else ''}",
                evaluation_type="api",
                global_step=global_step,
                epoch=epoch,
            )
            _save_evaluation_record(record, save_dir)

        return EvaluationResult(
            sample_idx=task.sample_idx,
            step_results=step_results,
            response_time=response_time,
        )

    except Exception as e:
        # ——> API 整体失败：同样不做任何缩放
        response_time = time.time() - start_time
        print(f"[parallel_eval] ❌ FAILED to evaluate sample {task.sample_idx}: {e}")

        uniform_flag = task.overall_adv > 0
        step_results = [uniform_flag for _ in task.steps]

        if save_dir:
            record = EvaluationRecord(
                sample_idx=task.sample_idx,
                query=task.query,
                rollout=task.rollout,
                steps=task.steps,
                overall_adv=task.overall_adv,
                llm_input_messages=[],
                llm_raw_output=f"ERROR: {str(e)}",
                llm_parsed_results=step_results,
                response_time=response_time,
                timestamp=time.time(),
                model_name=model_name,
                evaluation_type="api",
                global_step=global_step,
                epoch=epoch,
            )
            _save_evaluation_record(record, save_dir)

        return EvaluationResult(
            sample_idx=task.sample_idx,
            step_results=step_results,
            response_time=response_time,
        )

# ————————————————————————————————————————————————————————————————
# 统一的并行评估接口
# ————————————————————————————————————————————————————————————————

async def evaluate_step_flags_parallel(tokenizer, batch, model_name: str = "qwen-max", evaluation_type: Literal["api"] = "api", max_concurrent: int = 20, batch_size_limit: int = 100, mask_tensor: torch.Tensor = None, api_max_retries: int = 200, save_dir: Optional[str] = None, global_step: Optional[int] = None, epoch: Optional[str] = None) -> Tuple[List[List[bool]], Dict]:
    """并行评估step flags，每个sample一次API调用评估所有steps"""
    batch_size = len(batch.batch['prompts'])
    print(f"[parallel_eval] Starting parallel evaluation for {batch_size} samples using API mode")
    print(f"[parallel_eval] 🚀 OPTIMIZED: One API call per sample (not per step)")
    print(f"[parallel_eval] Model: {model_name}, API max retries: {api_max_retries}")
    if save_dir:
        print(f"[parallel_eval] Saving evaluation records to: {save_dir}")
    
    if 'steps' not in batch.non_tensor_batch:
        raise ValueError("❌ batch.non_tensor_batch['steps'] is required but not found")
    
    if evaluation_type != "api":
        raise ValueError(f"❌ Only 'api' evaluation_type is supported, got: {evaluation_type}")
    
    # 初始化API客户端
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ [parallel_eval] No API key found in DASHSCOPE_API_KEY environment variable")
        print("❌ [parallel_eval] Please set: export DASHSCOPE_API_KEY='your-api-key'")
        print("❌ [parallel_eval] Using random fallback for evaluation")
        return _apply_fallback_strategy_parallel(batch), {"fallback_used": True, "evaluation_type": evaluation_type}
    
    api_client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    # 🚀 关键优化：按sample创建任务，而不是按step
    all_tasks = []
    flags_per_sample = [[] for _ in range(batch_size)]
    skipped_samples = 0
    
    if mask_tensor is not None:
        response_mask = mask_tensor
        print(f"[parallel_eval] Using external mask tensor with shape {mask_tensor.shape}")
        
        response_length = batch.batch["responses"].size(1)
        if response_mask.shape != (batch_size, response_length):
            raise ValueError(f"❌ mask_tensor shape {response_mask.shape} doesn't match expected shape ({batch_size}, {response_length})")
    else:
        response_length = batch.batch["responses"].size(1)
        response_mask = batch.batch["loss_mask"][:, -response_length:]
        print(f"[parallel_eval] Using default loss_mask")

    for sample_idx in range(batch_size):
        query = tokenizer.decode(batch.batch["prompts"][sample_idx], skip_special_tokens=True)
        rollout = tokenizer.decode(batch.batch["responses"][sample_idx], skip_special_tokens=True)
        steps = batch.non_tensor_batch["steps"][sample_idx]
        
        sample_mask = response_mask[sample_idx]
        overall_adv = _get_overall_advantage(batch.batch["advantages"][sample_idx], sample_mask)
        
        if abs(overall_adv) < 1e-8:
            print(f"[parallel_eval] Sample {sample_idx}: advantage≈0 ({overall_adv:.6f}), skipping evaluation, returning all GOOD")
            flags_per_sample[sample_idx] = [True] * len(steps)
            skipped_samples += 1
            
            if save_dir:
                record = EvaluationRecord(
                    sample_idx=sample_idx,
                    query=query,
                    rollout=rollout,
                    steps=steps,
                    overall_adv=overall_adv,
                    llm_input_messages=[],
                    llm_raw_output="SKIPPED_ZERO_ADVANTAGE",
                    llm_parsed_results=[True] * len(steps),
                    response_time=0.0,
                    timestamp=time.time(),
                    model_name=model_name,
                    evaluation_type=evaluation_type,
                    global_step=global_step,
                    epoch=epoch
                )
                _save_evaluation_record(record, save_dir)
            continue
        
        # 🚀 关键优化：一个sample一个任务，而不是每个step一个任务
        task = EvaluationTask(
            sample_idx=sample_idx,
            query=query,
            rollout=rollout,
            steps=steps,
            overall_adv=overall_adv
        )
        all_tasks.append(task)
    
    total_tasks = len(all_tasks)
    total_api_calls = total_tasks  # 现在每个sample只需要一次API调用
    total_steps = sum(len(batch.non_tensor_batch["steps"][i]) for i in range(batch_size))
    
    print(f"[parallel_eval] 🚀 EFFICIENCY GAIN:")
    print(f"[parallel_eval]   - Total samples: {batch_size}")
    print(f"[parallel_eval]   - Total steps: {total_steps}")
    print(f"[parallel_eval]   - API calls needed: {total_api_calls} (instead of {total_steps})")
    print(f"[parallel_eval]   - Efficiency gain: {total_steps/max(1,total_api_calls):.1f}x fewer API calls")
    print(f"[parallel_eval]   - Skipped {skipped_samples} samples with advantage=0")
    
    if total_tasks == 0:
        print("[parallel_eval] No tasks to process, all samples had advantage=0")
        await api_client.close()
        return flags_per_sample, {
            "total_tasks": 0,
            "total_api_calls": 0,
            "total_steps": total_steps,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "total_api_time": 0,
            "avg_api_time": 0,
            "max_concurrent": max_concurrent,
            "fallback_used": False,
            "skipped_samples": skipped_samples,
            "evaluation_type": evaluation_type,
            "api_max_retries": api_max_retries,
            "efficiency_gain": 0
        }
    
    all_results = []
    semaphore = asyncio.Semaphore(max_concurrent)
    
    with tqdm(total=total_tasks, desc=f"[parallel_eval] Processing samples (API)") as pbar:
        for i in range(0, total_tasks, batch_size_limit):
            batch_tasks = all_tasks[i:i + batch_size_limit]
            
            # 每个task调用_evaluate_single_sample_api，一次性评估整个sample的所有steps
            coroutines = [
                _evaluate_single_sample_api(api_client, model_name, task, semaphore, api_max_retries, save_dir, global_step, epoch)
                for task in batch_tasks
            ]
            
            batch_results = await asyncio.gather(*coroutines, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    print(f"[parallel_eval] ❌ Task failed with exception: {result}")
                    continue
                all_results.append(result)
            
            pbar.update(len(batch_tasks))
    
    # 整理结果到flags_per_sample
    for result in all_results:
        flags_per_sample[result.sample_idx] = result.step_results
    
    total_time = sum(r.response_time for r in all_results)
    avg_time = total_time / len(all_results) if all_results else 0
    
    stats = {
        "total_tasks": total_tasks,
        "total_api_calls": len(all_results),
        "total_steps": total_steps,
        "successful_tasks": len(all_results),
        "failed_tasks": total_tasks - len(all_results),
        "total_api_time": total_time,
        "avg_api_time": avg_time,
        "max_concurrent": max_concurrent,
        "fallback_used": False,
        "skipped_samples": skipped_samples,
        "evaluation_type": evaluation_type,
        "model_name": model_name,
        "api_max_retries": api_max_retries,
        "save_dir": save_dir,
        "efficiency_gain": total_steps / max(1, len(all_results))  # 效率提升倍数
    }
    
    print(f"[parallel_eval] ✅ Completed with {stats['efficiency_gain']:.1f}x efficiency gain!")
    print(f"[parallel_eval] Stats: {stats}")
    
    await api_client.close()
    return flags_per_sample, stats

def _apply_fallback_strategy_parallel(batch) -> List[List[bool]]:
    """API 不可用时的并行 fallback：全部禁用缩放"""
    flags_per_sample = []
    advantages = batch.batch["advantages"]

    for sample_idx, steps in enumerate(batch.non_tensor_batch["steps"]):
        # 用 advantages 的符号来决定“无缩放”时应该返回 GOOD 还是 BAD
        overall_adv = _get_overall_advantage(advantages[sample_idx])
        uniform_flag = overall_adv > 0  # True=GOOD, False=BAD
        flags_per_sample.append([uniform_flag for _ in steps])

    return flags_per_sample

# ————————————————————————————————————————————————————————————————
# 向量化的mask应用
# ————————————————————————————————————————————————————————————————

def apply_step_mask_vectorized(batch, step_flags: List[List[bool]], consistent_scale: float = 1.0, pos_unconsistent_scale: float = 0.2, neg_unconsistent_scale: float = -0.2, mask_tensor: torch.Tensor = None) -> Dict:
    """向量化版本的step mask应用"""
    print(f"[vectorized_mask] Starting vectorized mask application")

    if 'step_ids' not in batch.batch:
        raise ValueError("❌ batch.batch['step_ids'] is required but not found")

    adv = batch.batch["advantages"]
    step_ids = batch.batch["step_ids"].to(adv.device)
    bs, resp_len = adv.shape

    if len(step_flags) != bs:
        raise ValueError(f"❌ step_flags length ({len(step_flags)}) != batch size ({bs})")

    if mask_tensor is not None:
        response_mask = mask_tensor
        print(f"[vectorized_mask] Using external mask tensor with shape {mask_tensor.shape}")
        if response_mask.shape != (bs, resp_len):
            raise ValueError(f"❌ mask_tensor shape {response_mask.shape} doesn't match expected shape ({bs}, {resp_len})")
    else:
        response_mask = batch.batch["loss_mask"][:, -resp_len:]
        print(f"[vectorized_mask] Using default loss_mask")

    overall_advs = []
    for sample_idx in range(bs):
        sample_mask = response_mask[sample_idx]
        overall_adv = _get_overall_advantage(adv[sample_idx], sample_mask)
        overall_advs.append(overall_adv)
    overall_advs = torch.tensor(overall_advs, device=adv.device)
    overall_pos = overall_advs > 0

    scale = torch.ones_like(adv)

    stats = {
        "total_samples": bs,
        "total_tokens": int(resp_len * bs),
        "tokens_modified": 0,
        "good_steps": 0,
        "bad_steps": 0,
        "positive_samples": int(overall_pos.sum().item()),
        "negative_samples": int((~overall_pos).sum().item()),
        "zero_adv_samples": 0,
        "pos_good_steps": 0,
        "pos_bad_steps": 0,
        "neg_good_steps": 0,
        "neg_bad_steps": 0,
        "pos_tokens": 0,
        "neg_tokens": 0,
    }

    for b in tqdm(range(bs), desc="[vectorized_mask] Processing samples"):
        current_step_flags = step_flags[b]
        overall_adv_sum = overall_advs[b].item()

        if abs(overall_adv_sum) < 1e-8:
            stats["zero_adv_samples"] += 1
            continue

        if not current_step_flags:
            continue

        sample_step_ids = step_ids[b]
        sample_overall_pos = bool(overall_pos[b].item())

        for step_id, is_good in enumerate(current_step_flags):
            step_mask = (sample_step_ids == step_id)
            if not step_mask.any():
                continue

            if sample_overall_pos:
                factor = consistent_scale if is_good else pos_unconsistent_scale
            else:
                factor = neg_unconsistent_scale if is_good else consistent_scale

            scale[b].masked_fill_(step_mask, factor)

            tokens_in_step = int(step_mask.sum().item())
            stats["tokens_modified"] += tokens_in_step

            if is_good:
                stats["good_steps"] += 1
                if sample_overall_pos:
                    stats["pos_good_steps"] += 1
                else:
                    stats["neg_good_steps"] += 1
            else:
                stats["bad_steps"] += 1
                if sample_overall_pos:
                    stats["pos_bad_steps"] += 1
                else:
                    stats["neg_bad_steps"] += 1

            if sample_overall_pos:
                stats["pos_tokens"] += tokens_in_step
            else:
                stats["neg_tokens"] += tokens_in_step

    padding_mask = (step_ids == -1)
    scale.masked_fill_(padding_mask, 1.0)

    original_adv_sum = adv.sum().item()
    batch.batch["advantages"] = adv * scale
    new_adv_sum = batch.batch["advantages"].sum().item()
    batch.batch["semantic_scale"] = scale

    valid_token_mask = response_mask & (~padding_mask)
    pos_token_mask = (adv > 0) & valid_token_mask
    neg_token_mask = (adv < 0) & valid_token_mask
    stats["pos_tokens_raw"] = int(pos_token_mask.sum().item())
    stats["neg_tokens_raw"] = int(neg_token_mask.sum().item())

    stats["original_adv_sum"] = original_adv_sum
    stats["new_adv_sum"] = new_adv_sum
    stats["adv_change_ratio"] = new_adv_sum / original_adv_sum if original_adv_sum != 0 else 1.0

    print(f"[vectorized_mask] Completed. Advantages: {original_adv_sum:.4f} -> {new_adv_sum:.4f}")
    print(f"[vectorized_mask] Modified {stats['tokens_modified']} tokens ({stats['good_steps']} good steps, {stats['bad_steps']} bad steps)")
    print(f"[vectorized_mask] Skipped {stats['zero_adv_samples']} samples with advantage=0")

    return stats

# ————————————————————————————————————————————————————————————————
# 统一的处理器类
# ————————————————————————————————————————————————————————————————

class ParallelSemanticProcessor:
    """并行语义处理器，只支持API评估"""
    
    def __init__(self, max_concurrent: int = 20, batch_size_limit: int = 100, model_name: str = "qwen-max", evaluation_type: Literal["api"] = "api", api_max_retries: int = 200):
        if evaluation_type != "api":
            raise ValueError(f"❌ Only 'api' evaluation_type is supported, got: {evaluation_type}")
            
        self.max_concurrent = max_concurrent
        self.batch_size_limit = batch_size_limit
        self.model_name = model_name
        self.evaluation_type = evaluation_type
        self.api_max_retries = api_max_retries
        
        print(f"[ParallelSemanticProcessor] 🚀 Initialized with OPTIMIZED evaluation (one API call per sample)")
        print(f"[ParallelSemanticProcessor] Settings: model={model_name}, concurrent={self.max_concurrent}, batch_limit={self.batch_size_limit}, api_retries={self.api_max_retries}")
        
    async def process_batch(self, tokenizer, batch, consistent_scale: float = 1.0, pos_unconsistent_scale: float = 0.2, neg_unconsistent_scale: float = -0.2, mask_tensor: torch.Tensor = None, save_dir: Optional[str] = None, global_step: Optional[int] = None, epoch: Optional[str] = None) -> Dict:
        """处理整个batch的语义评估和mask应用"""
        start_time = time.time()
        
        print(f"[ParallelSemanticProcessor] Starting OPTIMIZED step evaluation with API...")
        if save_dir:
            print(f"[ParallelSemanticProcessor] Evaluation records will be saved to: {save_dir}")
        eval_start = time.time()
        
        step_flags, eval_stats = await evaluate_step_flags_parallel(
            tokenizer=tokenizer,
            batch=batch,
            model_name=self.model_name,
            evaluation_type=self.evaluation_type,
            max_concurrent=self.max_concurrent,
            batch_size_limit=self.batch_size_limit,
            mask_tensor=mask_tensor,
            api_max_retries=self.api_max_retries,
            save_dir=save_dir,
            global_step=global_step,
            epoch=epoch
        )
        
        eval_time = time.time() - eval_start
        efficiency_gain = eval_stats.get('efficiency_gain', 1.0)
        print(f"[ParallelSemanticProcessor] ✅ Step evaluation completed in {eval_time:.2f}s with {efficiency_gain:.1f}x efficiency gain!")
        
        print("[ParallelSemanticProcessor] Applying step mask...")
        mask_start = time.time()
        
        mask_stats = apply_step_mask_vectorized(
            batch=batch,
            step_flags=step_flags,
            consistent_scale=consistent_scale,
            pos_unconsistent_scale=pos_unconsistent_scale,
            neg_unconsistent_scale=neg_unconsistent_scale,
            mask_tensor=mask_tensor
        )
        
        mask_time = time.time() - mask_start
        print(f"[ParallelSemanticProcessor] Step mask applied in {mask_time:.2f}s")
        
        total_time = time.time() - start_time
        
        combined_stats = {
            "total_processing_time": total_time,
            "evaluation_time": eval_time,
            "mask_application_time": mask_time,
            "evaluation_stats": eval_stats,
            "mask_stats": mask_stats,
            "speedup_info": {
                "parallel_evaluation": True,
                "vectorized_masking": True,
                "optimized_api_calls": True,
                "efficiency_gain": efficiency_gain,
                "max_concurrent": self.max_concurrent,
                "evaluation_type": self.evaluation_type,
                "model_name": self.model_name,
                "api_max_retries": self.api_max_retries,
                "save_dir": save_dir
            }
        }
        
        print(f"[ParallelSemanticProcessor] ✅ Total processing time: {total_time:.2f}s with {efficiency_gain:.1f}x API efficiency!")
        return combined_stats
    
    def process_batch_sync(self, tokenizer, batch, mask_tensor: torch.Tensor = None, save_dir: Optional[str] = None, global_step: Optional[int] = None, epoch: Optional[str] = None, **kwargs) -> Dict:
        """同步版本的batch处理"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.process_batch(tokenizer, batch, mask_tensor=mask_tensor, 
                             save_dir=save_dir, global_step=global_step, epoch=epoch, **kwargs)
        )

# ————————————————————————————————————————————————————————————————
# 同步包装函数
# ————————————————————————————————————————————————————————————————

def evaluate_step_flags(tokenizer, batch, good_words: tuple[str, ...] = ("GOOD",), bad_words: tuple[str, ...] = ("BAD",), model_name: str = "qwen-max", evaluation_type: Literal["api"] = "api", use_parallel: bool = True, max_concurrent: int = 20, mask_tensor: torch.Tensor = None, api_max_retries: int = 200, save_dir: Optional[str] = None, global_step: Optional[int] = None, epoch: Optional[str] = None) -> List[List[bool]]:
    """兼容性包装函数，只支持并行API评估"""
    if not use_parallel:
        raise ValueError("❌ Only parallel evaluation is supported")
    if evaluation_type != "api":
        raise ValueError(f"❌ Only 'api' evaluation_type is supported, got: {evaluation_type}")
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    flags, stats = loop.run_until_complete(
        evaluate_step_flags_parallel(
            tokenizer=tokenizer,
            batch=batch,
            model_name=model_name,
            evaluation_type=evaluation_type,
            max_concurrent=max_concurrent,
            mask_tensor=mask_tensor,
            api_max_retries=api_max_retries,
            save_dir=save_dir,
            global_step=global_step,
            epoch=epoch
        )
    )
    
    print(f"[evaluate_step_flags] ✅ Parallel execution completed with {stats.get('efficiency_gain', 1.0):.1f}x efficiency gain!")
    return flags

def apply_step_mask(batch, step_flags: List[List[bool]], consistent_scale: float = 1.0, pos_unconsistent_scale: float = 0.2, neg_unconsistent_scale: float = -0.2, use_vectorized: bool = True, mask_tensor: torch.Tensor = None):
    """兼容性包装函数，只支持向量化版本"""
    if not use_vectorized:
        raise ValueError("❌ Only vectorized version is supported")
    
    stats = apply_step_mask_vectorized(
        batch=batch,
        step_flags=step_flags,
        consistent_scale=consistent_scale,
        pos_unconsistent_scale=pos_unconsistent_scale,
        neg_unconsistent_scale=neg_unconsistent_scale,
        mask_tensor=mask_tensor
    )
    return stats