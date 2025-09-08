import copy
import time
from typing import Callable, Optional, Sequence
import uuid

from loguru import logger
from beyondagent.client.env_client import EnvClient
from beyondagent.module.agent_flow.agent_flow import AgentFlow
from beyondagent.module.agent_flow.base_agent_flow import BaseAgentFlow
from beyondagent.module.env_manager.env_worker import EnvWorker
from beyondagent.module.task_manager.agent_flow import ModifiedAgentFlow
from beyondagent.module.task_manager.base import LlmClient
from beyondagent.schema.task import Task, TaskObjective
from beyondagent.schema.trajectory import Trajectory
from beyondagent.module.task_manager.strategies.common.prompts.prompt_extract_refsol import (
    get_task_summarize_prompt,
    parse_tasks_from_response,
)

from . import TaskPostFilter

import copy
import time
import logging
from typing import Sequence, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logger = logging.getLogger(__name__)

class LlmFilter(TaskPostFilter):
    def __init__(self, env_url: str, llm_client: LlmClient, num_threads: int, *, tokenizer, config):
        self._env_client = EnvClient(env_url)
        self._llm_client = llm_client
        
        self._num_threads = num_threads
        
        self._tokenizer = tokenizer
        self._config = config
        
        # 用于线程安全的锁（如果需要共享资源访问控制）
        self._lock = threading.Lock()

    def filter(self, tasks: Sequence[TaskObjective]) -> list[TaskObjective]:
        """使用多线程并行过滤任务"""
        if not tasks:
            return []
        
        # 方法1: 使用 ThreadPoolExecutor 并行处理
        return self._filter_with_threadpool(tasks)
        
        # 方法2: 如果需要更细粒度的控制，可以使用下面的批处理方法
        # return self._filter_with_batches(tasks)
    
    def _filter_with_threadpool(self, tasks: Sequence[TaskObjective]) -> list[TaskObjective]:
        """使用线程池并行处理所有任务"""
        res = []
        
        with ThreadPoolExecutor(max_workers=self._num_threads) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(self._execute_strategy1, task): task 
                for task in tasks
            }
            
            # 收集结果
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    t=future.result()
                    if t is not None:
                        res.append(t)
                except Exception as e:
                    logger.exception(f"Error processing task {task}: {e}")
                    # 根据需求决定是否将失败的任务也包含在结果中
                    # 这里选择跳过失败的任务
                    continue
        
        return res
    
    def _filter_with_batches(self, tasks: Sequence[TaskObjective]) -> list[TaskObjective]:
        """分批处理任务（可选的实现方式）"""
        res = []
        tasks_list = list(tasks)
        
        # 计算批次大小
        batch_size = max(1, len(tasks_list) // self._num_threads)
        
        # 分批处理
        for i in range(0, len(tasks_list), batch_size):
            batch = tasks_list[i:i + batch_size]
            
            with ThreadPoolExecutor(max_workers=min(self._num_threads, len(batch))) as executor:
                future_to_task = {
                    executor.submit(self._execute_strategy1, task): task 
                    for task in batch
                }
                
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        if future.result():
                            res.append(task)
                    except Exception as e:
                        logger.exception(f"Error processing task {task}: {e}")
                        continue
        
        return res
    
    def _execute_strategy1(self, task: TaskObjective) -> TaskObjective|None:
        """Execute strategy 1: Simple execution / 执行策略1：简单执行"""
        try:
            worker = EnvWorker(
                task.task,
                config=self._config,
                thread_index=0,
                tokenizer=self._tokenizer
            )
            agent_flow = ModifiedAgentFlow(
                enable_context_generator=False,
                llm_chat_fn=self._get_llm_chat_fn(),
                tokenizer=self._tokenizer,
                config=self._config,
            )
            assert task.objective is not None, "synthetic data must have objective"
            assert task.ground_truth is not None, "synthetic data must have ground-truth"
            traj = worker.execute(
                data_id="unknown",
                rollout_id="unknown",
                add_exp=False,
                task_train_exp_mode='woexp',
                agent_flow=agent_flow,
                tmux={
                    'step':[0],
                    'token':[0],
                },
                stop=[False], # 这俩玩意有没有什么办法能封装一下
                system_prompt=make_solver_tip_prompt(task.objective,task.ground_truth)
            )
            
            valid=self._validate(task,traj)
            if valid:
                task=self._rewrite_new_gt(task,traj)
                return task
            else:
                return None
        except Exception as e:
            logger.exception(f"Error in _execute_strategy1 for task {task}: {e}")
            return None
        
        
    def _validate(self, task: TaskObjective, trajectory: Trajectory) -> bool:
        """验证任务执行结果"""
        try:
            validator = TrajectoryEvaluator(self._llm_client)
            return validator.evaluate_trajectory_success(task, trajectory)
        except Exception as e:
            logger.exception(f"Error in _validate for task {task}: {e}")
            return False
    
    def _rewrite_new_gt(self,task:TaskObjective,trajectory:Trajectory)->TaskObjective:
        """重写任务新的ground-truth"""
        sys_prompt,user_prompt=get_task_summarize_prompt([trajectory],[task], None)
        messages=[{"role":"system","content":sys_prompt},{"role":"user","content":user_prompt}]
        llm_output=self._get_llm_chat_fn()(messages)["content"]
        gt = parse_tasks_from_response(llm_output)
        if gt is not None:
            task.ground_truth=gt
        
        return task        
        
    
    def _get_llm_chat_fn(self, sampling_params: Optional[dict] = None) -> Callable:
        """获取LLM聊天函数，线程安全"""
        def llm_chat(
            messages: list[dict[str, str]],
            custom_sampling_params: Optional[dict] = None,
            request_id: Optional[str] = None,
        ) -> dict:
            """
            input messages: [{"role": "system", "value": "..."}, {"role": "user", "value": "..."}]
            output messages: [{"role": "assistant", "value": "..."}]
            """
            updated_sampling_params = {}
            if sampling_params:
                updated_sampling_params.update(sampling_params)
            if custom_sampling_params:
                updated_sampling_params.update(custom_sampling_params)

            input_messages = copy.deepcopy(messages)
            res = None
            
            # 重试机制
            for i in range(3):
                try:
                    # 如果 llm_client 不是线程安全的，可以在这里加锁
                    # with self._lock:
                    res = self._llm_client.chat(
                        messages=input_messages, 
                        sampling_params=updated_sampling_params
                    )
                    break
                except Exception as e:
                    logger.exception(f"llm_chat attempt {i+1} error: {e}")
                    time.sleep(10*2**i)

            assert res is not None, f"LLM client failed to chat after 3 attempts"
            return {
                "role": "assistant",
                "content": res,
            }

        return llm_chat




from typing import List


class TrajectoryEvaluator:
    """Evaluate trajectory success using LLM / 使用LLM评估轨迹成功"""
    
    def __init__(self, client:LlmClient):
        self.client = client
        self.prompts = EvaluationPrompts()
    
    def evaluate_trajectory_success(self, task:TaskObjective, trajectory:Trajectory) -> bool:
        """Evaluate if trajectory completed the task successfully / 评估轨迹是否成功完成任务"""
        try:
            # Create trajectory summary / 创建轨迹摘要
            trajectory_summary = self._create_trajectory_summary(trajectory)
            
            final_observation: str|None = None
            for step in reversed(trajectory.steps):
                if final_observation is None and step['role']!='assistant':
                    final_observation = step['content']
                    break
                
            # Generate evaluation prompt / 生成评估提示
            assert task.objective is not None, "synthetic task must have objective"
            prompt = self.prompts.success_evaluation_prompt(
                query=task.objective,
                trajectory_summary=trajectory_summary,
                final_observation=final_observation or "[no observation]"
            )
            
            # Get LLM evaluation / 获取LLM评估
            response = self.client.chat(prompt,sampling_params={})
            
            # Parse evaluation result / 解析评估结果
            success = self._parse_evaluation_response(response)
            
            logger.debug(f"Trajectory evaluation result: {success}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to evaluate trajectory: {e}")
            return False
    
    def _create_trajectory_summary(self, traj: Trajectory) -> str:
        """Create summary of trajectory steps / 创建轨迹步骤摘要"""
        summary_blocks = []
        
        for i, step in enumerate(traj.steps):
            block=f"(Step {i+1}) {step['role']}:\n"
            block+=f"{step['content'][:200]}...\n"
            summary_blocks.append(block)
        
        return "\n".join(summary_blocks)
    
    def _parse_evaluation_response(self, response: str) -> bool:
        """Parse LLM evaluation response / 解析LLM评估响应"""
        if not response:
            return False
        
        response_lower = response.lower()
        
        # Look for explicit success/failure indicators / 查找明确的成功/失败指标
        if 'success: true' in response_lower or 'successful: true' in response_lower:
            return True
        elif 'success: false' in response_lower or 'successful: false' in response_lower:
            return False
        
        # Look for keywords / 查找关键词
        success_keywords = ['success', 'completed', 'achieved', 'accomplished', 'solved']
        failure_keywords = ['failed', 'incomplete', 'unsuccessful', 'not completed', 'not achieved']
        
        success_count = sum(1 for keyword in success_keywords if keyword in response_lower)
        failure_count = sum(1 for keyword in failure_keywords if keyword in response_lower)
        
        # Default to success if more success keywords found / 如果找到更多成功关键词则默认成功
        return success_count > failure_count
    
    

class EvaluationPrompts:
    """Prompt templates for trajectory evaluation / 轨迹评估的提示模板"""
    
    def success_evaluation_prompt(self, query: str, trajectory_summary: str,
                                final_observation: str) -> list:
        """Prompt for evaluating trajectory success / 评估轨迹成功的提示"""

# - Expected Outcome (Ground Truth API Call or Result): {ground_truth}
        messages = [
            {
            "role": "user",
            "content": f"""You are a strict task evaluation expert. Your goal is to determine whether the following multi-step agent trajectory successfully completed the assigned task.

    # Task Details
    - Query: {query}

    # Execution Summary
    - Trajectory Summary:
    {trajectory_summary}

    - Final Observation: {final_observation}

    # Evaluation Instructions

    Carefully analyze the trajectory to determine if the task was truly completed. Specifically, consider the following aspects:

    1. **API Matching**: Did the agent correctly call the required APIs according to the task requirements?
    2. **Parameter Usage**: Were the parameters used in API calls correct and sufficient?
    3. **Logical Flow**: Was the sequence of steps logical without unreasonable skips?
    4. **Final Result**: Did the final state achieve the expected outcome, reasonably solve the task, obtain all necessary information, and complete the task objectives?
    5. **Failed or Skipped Steps**: Were there any critical errors, skipped steps, or invalid code that prevented the task from being actually executed?

    # Format Your Response Strictly As:

    Success: [true/false]
    Reason: [Concise and specific explanation, referring to the above criteria.]

    Note: Do NOT mark the task as successful if the correct API was never called, the parameters were incorrect, or the result was not achieved, even if the intent seemed right.
    """
            }
        ]
        
        return messages



def make_solver_tip_prompt(query: str, gt:str):
    return f"""You are an AI assistant helping to complete tasks in an interactive environment.
Feel free to use the tips to help you complete the task. The tips include a potential step-by-step solution to the task, but I do not ensure it is correct.

Your Query:
{query}

Tips:
{gt}
"""