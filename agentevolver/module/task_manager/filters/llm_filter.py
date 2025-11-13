import copy
import time
from typing import Callable, Optional, Sequence
import uuid

from loguru import logger
from agentevolver.client.env_client import EnvClient
from agentevolver.module.agent_flow.agent_flow import AgentFlow
from agentevolver.module.agent_flow.base_agent_flow import BaseAgentFlow
from agentevolver.module.env_manager.env_worker import EnvWorker
from agentevolver.module.exp_manager.exp_manager import TrajExpConfig
from agentevolver.module.task_manager.agent_flow import ModifiedAgentFlow
from agentevolver.module.task_manager.base import LlmClient
from agentevolver.schema.task import Task, TaskObjective
from agentevolver.schema.trajectory import Trajectory
from agentevolver.module.task_manager.strategies.common.prompts.prompt_extract_refsol import (
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
        """
        Initializes the LlmFilter with the necessary components for filtering tasks.

        Args:
            env_url (str): The URL of the environment client.
            llm_client (LlmClient): The Language Model client used for task validation and rewriting.
            num_threads (int): The number of threads to use for parallel processing.
            tokenizer: The tokenizer used for text processing.
            config: Configuration settings for the filter.
        """
        self._env_client = EnvClient(env_url)
        self._llm_client = llm_client
        
        self._num_threads = num_threads
        
        self._tokenizer = tokenizer
        self._config = config
        
        self._lock = threading.Lock()  # ⭐ Initialize a lock for thread-safe operations

    def filter(self, tasks: Sequence[TaskObjective]) -> list[TaskObjective]:
        """
        Filters and processes a sequence of tasks using multi-threading.

        Args:
            tasks (Sequence[TaskObjective]): The sequence of tasks to be filtered.

        Returns:
            list[TaskObjective]: The filtered and processed tasks.
        """
        if not tasks:
            return []
        
        return self._filter_with_threadpool(tasks)
        
        # Optional method 2: Process tasks in batches
        # return self._filter_with_batches(tasks)

    def _filter_with_threadpool(self, tasks: Sequence[TaskObjective]) -> list[TaskObjective]:
        """use thread pool to handle tasks parallelly"""
        from tqdm import tqdm
        res = []
        
        progress = tqdm(total=len(tasks), desc="Filtering tasks")
        with ThreadPoolExecutor(max_workers=self._num_threads) as executor:
            # submit all tasks
            future_to_task = {
                executor.submit(self._execute_strategy1, task): task 
                for task in tasks
            }
            
            # collect results
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                progress.update(1)
                try:
                    t = future.result()
                    if t is not None:
                        res.append(t)
                except Exception as e:
                    logger.exception(f"Error processing task {task}: {e}")
                    # ignore the failed tasks
                    continue
        progress.close()
        return res
    
    def _filter_with_batches(self, tasks: Sequence[TaskObjective]) -> list[TaskObjective]:
        """
        Filters and processes a sequence of tasks in batches using multi-threading.

        Args:
            tasks (Sequence[TaskObjective]): A sequence of TaskObjective objects to be processed.

        Returns:
            list[TaskObjective]: A list of TaskObjective objects that passed the filtering criteria.
        """
        res = []
        tasks_list = list(tasks)
        
        # Calculate the batch size
        batch_size = max(1, len(tasks_list) // self._num_threads)  # ⭐ Ensures at least one task per batch and distributes tasks evenly across threads
        
        # Process tasks in batches
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

    def _execute_strategy1(self, task: TaskObjective) -> TaskObjective | None:
        """
        Executes the first strategy for processing a task. This includes setting up a worker and an agent flow,
        executing the task, validating the result, and potentially rewriting the task's ground truth if the
        execution is valid.

        Args:
            task (TaskObjective): The task to be processed.

        Returns:
            TaskObjective|None: The updated task if the execution is valid, otherwise None.
        """
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
                traj_exp_config=TrajExpConfig(add_exp=False),
                agent_flow=agent_flow,
                tmux={
                    'step': [0],
                    'token': [0],
                },
                stop=[False],  # parameter tmux and stop could be refactored
                system_prompt=make_solver_tip_prompt(task.objective, task.ground_truth)
            )  # ⭐ Execute the task using the worker and agent flow

            valid = self._validate(task, traj)
            if valid:
                task = self._rewrite_new_gt(task, traj)
                return task
            else:
                return None
        except Exception as e:
            logger.exception(f"Error in _execute_strategy1 for task {task}: {e}")
            return None


    def _validate(self, task: TaskObjective, trajectory: Trajectory) -> bool:
        """
        Validates the success of a task's execution by evaluating the provided trajectory.

        Args:
            task (TaskObjective): The task objective to be validated.
            trajectory (Trajectory): The trajectory of the task execution.

        Returns:
            bool: True if the task was successfully executed, False otherwise.
        """
        try:
            validator = TrajectoryEvaluator(self._llm_client)
            return validator.evaluate_trajectory_success(task, trajectory)  # ⭐ Core line that evaluates the success of the task
        except Exception as e:
            logger.exception(f"Error in _validate for task {task}: {e}")
            return False

    def _rewrite_new_gt(self, task: TaskObjective, trajectory: Trajectory) -> TaskObjective:
        """
        Rewrite ground-truth
        """
        sys_prompt, user_prompt = get_task_summarize_prompt([trajectory], [task], None)
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
        llm_output = self._get_llm_chat_fn()(messages)["content"]
        gt = parse_tasks_from_response(llm_output)
        if gt is not None:
            task.ground_truth = gt
        
        return task        
        
    
    def _get_llm_chat_fn(self, sampling_params: Optional[dict] = None) -> Callable:
        """
        Returns a thread-safe LLM chat function.

        Args:
            sampling_params (Optional[dict]): Default sampling parameters for the LLM.

        Returns:
            Callable: A function that can be used to chat with the LLM.
        """
        def llm_chat(
            messages: list[dict[str, str]],
            custom_sampling_params: Optional[dict] = None,
            request_id: Optional[str] = None,
        ) -> dict:
            """
            Sends input messages to the LLM and returns the assistant's response.

            Args:
                messages (list[dict[str, str]]): List of message dictionaries, each containing 'role' and 'value'.
                custom_sampling_params (Optional[dict]): Custom sampling parameters to override or add to the default ones.
                request_id (Optional[str]): An optional request ID for tracking.

            Returns:
                dict: A dictionary containing the assistant's role and content.
            """
            updated_sampling_params = {}
            if sampling_params:
                updated_sampling_params.update(sampling_params)
            if custom_sampling_params:
                updated_sampling_params.update(custom_sampling_params)

            input_messages = copy.deepcopy(messages)
            res = None
            
            # Retry mechanism
            for i in range(3):
                try:
                    # If llm_client is not thread-safe, a lock should be added here
                    # with self._lock:
                    res = self._llm_client.chat(
                        messages=input_messages, 
                        sampling_params=updated_sampling_params
                    )  # ⭐ Attempt to get a response from the LLM
                    break
                except Exception as e:
                    logger.exception(f"llm_chat attempt {i + 1} error: {e}")
                    time.sleep(10 * 2**i)

            assert res is not None, f"LLM client failed to chat after 3 attempts"
            return {
                "role": "assistant",
                "content": res,
            }

        return llm_chat




from typing import List


class TrajectoryEvaluator:
    """Evaluate trajectory success using LLM"""
    
    def __init__(self, client: LlmClient):
        self.client = client
        self.prompts = EvaluationPrompts()

    def evaluate_trajectory_success(self, task: TaskObjective, trajectory: Trajectory) -> bool:
        """Evaluate if trajectory completed the task successfully"""
        try:
            # Create trajectory summary
            trajectory_summary = self._create_trajectory_summary(trajectory)
            
            final_observation: str | None = None
            for step in reversed(trajectory.steps):
                if final_observation is None and step['role'] != 'assistant':
                    final_observation = step['content']
                    break
                
            # Generate evaluation prompt
            assert task.objective is not None, "synthetic task must have objective"
            prompt = self.prompts.success_evaluation_prompt(
                query=task.objective,
                trajectory_summary=trajectory_summary,
                final_observation=final_observation or "[no observation]"
            )
            
            # Get LLM evaluation
            response = self.client.chat(prompt, sampling_params={})
            
            # Parse evaluation result
            success = self._parse_evaluation_response(response)
            
            logger.debug(f"Trajectory evaluation result: {success}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to evaluate trajectory: {e}")
            return False
    
    def _create_trajectory_summary(self, traj: Trajectory) -> str:
        """
        Creates a summary of the steps in a given trajectory, with each step's content truncated to 200 characters.

        Args:
            traj (Trajectory): The trajectory object containing the steps to be summarized.

        Returns:
            str: A string summary of the trajectory steps.
        """
        summary_blocks = []
        
        for i, step in enumerate(traj.steps):
            block = f"(Step {i + 1}) {step['role']}:\n"
            block += f"{step['content'][:200]}...\n"  # ⭐ Truncate the content to 200 characters and append an ellipsis
            summary_blocks.append(block)
        
        return "\n".join(summary_blocks)
    
    def _parse_evaluation_response(self, response: str) -> bool:
        """
        Parses the evaluation response from an LLM to determine if the task was successful or not.

        Args:
            response (str): The response from the LLM.

        Returns:
            bool: True if the task is deemed successful, False otherwise.
        """
        if not response:
            return False
        
        response_lower = response.lower()
        
        # Look for explicit success/failure indicators
        if 'success: true' in response_lower or 'successful: true' in response_lower:
            return True
        elif 'success: false' in response_lower or 'successful: false' in response_lower:
            return False
        
        # Look for keywords
        success_keywords = ['success', 'completed', 'achieved', 'accomplished', 'solved']
        failure_keywords = ['failed', 'incomplete', 'unsuccessful', 'not completed', 'not achieved']
        
        success_count = sum(1 for keyword in success_keywords if keyword in response_lower)
        failure_count = sum(1 for keyword in failure_keywords if keyword in response_lower)
        
        # Default to success if more success keywords found
        return success_count > failure_count  # ⭐ Determine success based on the count of success and failure keywords
    
    

class EvaluationPrompts:
    """Prompt templates for trajectory evaluation"""
    
    def success_evaluation_prompt(self, query: str, trajectory_summary: str,
                                final_observation: str) -> list:
        """
        Generates a prompt for evaluating the success of a trajectory.

        Args:
            query (str): The task query.
            trajectory_summary (str): The summary of the trajectory.
            final_observation (str): The final observation of the trajectory.

        Returns:
            list: A list containing a single dictionary with the role and content for the prompt.
        """

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



def make_solver_tip_prompt(query: str, gt: str):
    return f"""You are an AI assistant helping to complete tasks in an interactive environment.
Feel free to use the tips to help you complete the task. The tips include a potential step-by-step solution to the task, but I do not ensure it is correct.

Your Query:
{query}

Tips:
{gt}
"""