import re
import threading
from typing import Any, Optional, cast
from loguru import logger
from agentevolver.client.env_client import EnvClient
from agentevolver.client.llm_client import DashScopeClient
from agentevolver.module.agent_flow.reward_calculator import GraderResult, RewardCalculator
from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory

from . import grader_manager

USER_PROMPT = """### Role
You are an expert AI agent evaluator. Your job is to judge an agent's performance using the following inputs:

1) **User Task** — what the agent was supposed to accomplish.  
2) **Reference Solution** — a correct approach/outcome to compare against (other valid solutions may exist).  
3) **Agent Trajectory** — chronological steps the agent took, including actions, decisions, and outputs.

### Ground Rules
- Base your judgment strictly on the provided trajectory. Do **not** invent missing steps or assumptions.
- Treat the Reference Solution as an oracle for correctness checks and efficiency comparison, while allowing alternative correct methods.
- When citing issues, reference concrete steps or observations from the trajectory.
- Be deterministic: follow the procedure below and the scoring constraints exactly.
- “Infinite or runaway repetition” means the agent repeats essentially the same step/loop ≥3 times with no new information or progress.

---

## Evaluation Procedure

**Step 1 — Relevance Gate (0 or proceed)**  
- Determine if the trajectory's steps are **materially related** to the User Task.  
- If the approach is wholly unrelated → **score = 0** and stop.  
- Otherwise, continue.

**Step 2 — Repetition Penalty Gate**  
- Check for infinite/runaway repetition of identical or near-identical steps.  
  - If such repetition exists:  
    - If steps are otherwise relevant → **final score must be ≤ 20**.  
    - If steps are irrelevant → **score = 0**.  
- If no infinite repetition, continue.

**Step 3 — Goal Achievement (Critical Binary Check)**  
- Examine **all** steps and the final result to decide if the task is actually completed **correctly**.  
- **Compare** both the final answer **and** the solution path against the Reference Solution to validate correctness.  
- Do not be misled by confident language—verify substance.

**Critic Details**
There are some critic details you should check:
- Some APIs are paginated, which is documented in the API doc. Agent must call the API multiple times to get all the data.

If agent does not consider these details, it may be wrong.

**Step 4 — Additional Deductions (respect the above ranges)**
- **Code Execution Errors:** Deduct for crashes, runtime errors, failed tool calls, or obvious bugs.
- **Efficiency & Conciseness vs. Reference:** If the trajectory is significantly more roundabout, redundant, or cluttered than the reference approach, deduct accordingly—even if correct. Unnecessary/irrelevant steps count here.

---

## Scoring Guidelines (choose a range, then adjust within it)
**If goal achieved (must be 60-100):**
- **90-100:** Exceptional — clean, efficient, equal/better than reference; no significant issues.
- **80-89:** Strong — correct with minor inefficiencies or small issues vs. the reference.
- **70-79:** Good — correct but notably less efficient or with several unnecessary steps.
- **60-69:** Adequate — correct yet with significant problems in efficiency, clarity, or execution quality.

**If goal not achieved (must be 0-40):**
- **30-40:** Poor — incorrect but generally relevant with partial progress aligned to the reference path.
- **10-29:** Very poor — incorrect with major execution issues; only weak alignment to a correct path.
- **1-9:** Minimal relevant attempt — incorrect with severe problems, but some faint relevance.
- **0:** Complete failure — irrelevant approach **or** infinite repetition of irrelevant steps.

> Note on Step 2 cap: If infinite/runaway repetition is detected and steps are otherwise relevant, the **maximum** final score is **20** (within the 0-40 band).

---

## Output Format
First, provide a **detailed reasoning analysis** that references specific steps/observations and compares against the Reference Solution (including efficiency notes and any code/error findings).
Then output a single integer score (either **0-40** or **60-100**, never 41-59) wrapped in tags:

<reward>75</reward>

---

** User Task **
{task}

** Reference Solution **
{reference_trajs}

** Agent Trajectory (STEP-ACTION-OBSERVATION) **
{trajs}


---
"""

USER_PROMPT_WITH_MEAN_CONSTRAINT=USER_PROMPT+"""
Over the past period of time, the average score you gave to some samples was {running_mean:.4f}.
Please note that the average score must be maintained around {mean_score:.4f} (+-0.2), or you will be penalized.
"""

def steps_to_msg(steps: list[dict[str, Any]]) -> str:
    """
    Converts a list of step dictionaries, each containing a role ('assistant' or 'user') and content, into a formatted string that represents the entire trajectory in a structured manner.

    Args:
        steps (list[dict[str, Any]]): A list of dictionaries, where each dictionary contains the role (either 'assistant' or 'user') and the content of the message.

    Returns:
        str: A formatted string representing the entire trajectory.
    """
    trajectory_text = ""
    assert steps[0]['role'] == 'assistant'
    for i, msg in enumerate(steps):
        role = msg.get("role", "unknown")
        if role == 'assistant':
            block = f""">>> STEP {i//2} <<<
<|ACTION|>
{msg['content']}
<|END|>
"""
        elif role == "user":
            block = f"""<|OBSERVATION|>
{msg['content']}
<|END|>
"""
        else:
            raise ValueError("roles in trajectory must be assistant or user")
        trajectory_text += block.strip() + "\n\n"  # ⭐ Append the formatted block to the trajectory text
    return trajectory_text

@grader_manager.reg("llm-binary")
class LlmAsJudgeBinaryRewardCalculator(RewardCalculator):
    """
    RewardCalculator that uses LLM as judge.
    """
    _running_judge_mean_fast = 0.3
    _running_judge_mean_slow = 0.3

    _alpha_fast=0.9
    _alpha_slow=0.95
    _update_lock = threading.Lock()

    def __init__(self, task: Task, model_name='qwen3-235b-a22b-instruct-2507', use_mean_constraint=True):
        """
        Initializes the reward calculator with a specific task, model name, and whether to use mean constraint.

        Args:
            task (Task): The task for which the reward is being calculated.
            model_name (str, optional): The name of the language model to be used as the judge. Defaults to 'qwq-plus'.
            use_mean_constraint (bool, optional): Whether to enforce a constraint on the mean score. Defaults to True.
        """
        super().__init__(task)
        
        self._client = DashScopeClient(model_name=model_name)
        self._client.max_tokens=32768
        self._use_mean_constraint = use_mean_constraint

    @classmethod
    def update_running_mean(cls, new_score: float):
        """
        Updates the running mean of the judge scores using an exponential moving average (EMA) for both a fast and slow decay.
        Ensures thread safety by using a lock.

        Args:
            new_score (float): The new score to be included in the running mean calculation.

        Returns:
            None
        """
        with cls._update_lock:  # ⭐ Acquire the lock to ensure thread safety
            cls._running_judge_mean_fast = cls._alpha_fast * cls._running_judge_mean_fast + (1-cls._alpha_fast) * new_score
            cls._running_judge_mean_slow = cls._alpha_slow * cls._running_judge_mean_slow + (1-cls._alpha_slow) * new_score

    @classmethod
    def get_running_mean(cls):
        """
        Returns the current running mean of the judge's scores.

        Returns:
            float: The current running mean.
        """
        with cls._update_lock:  # ⭐ Ensures thread safety when accessing the running mean
            return cls._running_judge_mean_fast

    @classmethod
    def get_stable_mean(cls):
        """
        Returns the current stable mean of the judge's scores.

        Returns:
            float: The current stable mean.
        """
        with cls._update_lock:  # ⭐ Ensures thread safety when accessing the stable mean
            return cls._running_judge_mean_slow

    def pack_message(self, trajectory: Trajectory):
        """
        Packs the given trajectory into a message format.

        Args:
            trajectory (Trajectory): The trajectory to be packed into a message.

        Returns:
            list: A list containing the packed message.
        """
        messages = []

        assert len(trajectory.steps) >= 2 and trajectory.steps[1]['role'] == 'user', "trajectory must start with system message and then user message"
        task_query = trajectory.steps[1]['content']
        
        assert self.task.ground_truth is not None, "ground truth must not be None for synthetic task"
        if self._use_mean_constraint:
            content=USER_PROMPT_WITH_MEAN_CONSTRAINT.format(
                task=task_query,
                trajs=steps_to_msg(trajectory.steps[2:]),
                running_mean=self.get_running_mean(),
                mean_score=self.get_stable_mean(),
                reference_trajs="[No solution provided, please judge the task by yourself]"
            )
        else:
            content=USER_PROMPT.format(
                task=task_query,
                trajs=steps_to_msg(trajectory.steps[2:]),
                reference_trajs="[No solution provided, please judge the task by yourself]"
            )
        messages.append(
            {
                "role": "user",
                "content": content  # ⭐ Constructs the content of the message based on the trajectory and other parameters
            }
        )
        return messages

    def calculate_reward(self, trajectory: Trajectory, env: EnvClient, instance_id: str) -> GraderResult:
        x,res = cast(tuple[float,str], self._calculate_reward(trajectory, env, eject_llm_output=True))
        return {
            "score": x,
            "reason": res
        }


    def _calculate_reward(self, trajectory: Trajectory, env: EnvClient, *, eject_llm_output: bool = False):
        """
        Evaluates a given trajectory in a specific environment by interacting with a language model (LLM) to get a response,
        then parses this response to calculate a reward score. The score is based on whether there was a critical failure,
        the intent comprehension, and the correctness of the completion. If `eject_llm_output` is set to `True`, it also
        returns the raw LLM response.

        Args:
            trajectory (Trajectory): The trajectory to evaluate.
            env (EnvClient): The environment where the trajectory is executed.
            eject_llm_output (bool, optional): Whether to return the raw LLM response. Defaults to False.

        Returns:
            float or tuple: The calculated reward score, and optionally the raw LLM response if `eject_llm_output` is True.
        """
        response=""
        for chunk in self._client.chat_stream_with_retry(messages=self.pack_message(trajectory),max_retries=64):
            response += chunk
        if response:
            import re
            reward_match = re.search(r'<reward>([\d\.]+)</reward>', response.strip())
            if reward_match:
                score = float(reward_match.group(1))
                score = max(0.0, min(100.0, score))/100.0
            else:
                print(f"Could not parse score from response: {response}")
                score=0.0
        else:
            print("No response from evaluation API")
            score=0.0
        
        if not eject_llm_output:
            return score
        else:
            return score,response

@grader_manager.reg("llm-binary-no_constraint")
class LlmAsJudgeBinaryRewardCalculatorNoConstraint(LlmAsJudgeBinaryRewardCalculator):
    def __init__(self, task: Task, model_name='qwen3-235b-a22b-instruct-2507'):
        super().__init__(task, model_name, use_mean_constraint=False)