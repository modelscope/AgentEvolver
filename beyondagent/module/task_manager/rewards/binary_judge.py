import re
import threading
from typing import Any, Optional, cast
from loguru import logger
from beyondagent.client.env_client import EnvClient
from beyondagent.client.llm_client import DashScopeClient
from beyondagent.module.agent_flow.reward_calculator import GraderResult, RewardCalculator
from beyondagent.schema.task import Task
from beyondagent.schema.trajectory import Trajectory

from . import grader_manager

USER_PROMPT = """
### Role Description
You are an expert AI agent evaluator. Your task is to assess an agent's performance based on its action trajectory and the original user request. Apply strict binary scoring (0/1) for each dimension without partial credits.

### Input Analysis
You will receive two inputs:
1. User Task: The original objective the agent should accomplish
2. Agent Trajectory: Sequential record of actions, decisions, and outputs during task execution

### Evaluation Procedure
Follow these steps sequentially:

#### Step 1: Critical Failure Check
Immediately score both dimensions 0 if ANY of these occur:
- All content is totally gibberish/unreadable
- Enters infinite loop or identical step repetition
- Completely irrelevant to user task
- Fails to produce any actionable output

#### Step 2: Task Intent Comprehension (Score 0 or 1)
- Score 1 ONLY if: 
  Agent accurately identifies core objective
  Initial actions align with task purpose
- Score 0 if:
  Misinterprets fundamental task purpose
  Shows contradictory understanding

#### Step 3: Task Correct Completion (Score 0 or 1)
Score 1 ONLY if ALL conditions are met:
- Step is logically valid and necessary. (Error is allowed in intermediate steps)
- Zero hallucinated information. (For example, fabricated information, misinterpreted fields are hallucinated.)
- Final output resolves user's request, or user's request is unrealistic and best effort is done (Check this by your own knowledge). 
Score 0 if ANY condition fails

### Mandatory Constraints
- Never combine scores or calculate totals
- Critical failure overrides all other checks
- Scores are independent (e.g., may score 1,0)

### Output
**Strictly follow this sequence:**
1. Perform Step 1 → Step 2 → Step 3 evaluations in order
2. Generate analysis covering all evaluation steps
3. Finaly output the evaluation result with the following FORMAT:
Reason: [Reason for score]
Critical Failure: [Yes/No]  
Intent Comprehension: [0/1]
Correct Completion: [0/1]

** User Task **:
{task}

** Agent Trajectory (STEP-ACTION-OBSERVATION) **:
{trajs}

** Reminder **:
Perform evaluation steps sequentially before generating output.
"""

USER_PROMPT_WITH_MEAN_CONSTRAINT = USER_PROMPT+"""
Over the past period of time, the average score you gave to some samples was {running_mean:.4f}.
Please note that the average score must be maintained around {mean_score:.4f} (+-0.2), or you will be penalized.
"""

def steps_to_msg(steps: list[dict[str, Any]]) -> str:
    # 添加轨迹消息（将所有对话转换为一个连贯的文本）
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
        trajectory_text += block.strip() + "\n\n"
    return trajectory_text



@grader_manager.reg("llm-binary")
class LlmAsJudgeBinaryRewardCalculator(RewardCalculator):
    """
    RewardCalculator that uses LLM as judge.
    """
    # 定义类变量，跨实例共享
    _running_judge_mean_fast = 0.3  # 初始化为默认值
    _running_judge_mean_slow = 0.3  # 初始化为默认值
    
    _alpha_fast=0.9
    _alpha_slow=0.95
    _update_lock = threading.Lock()  # 锁也需要作为类变量共享

    def __init__(self, task: Task, model_name='qwq-plus', use_mean_constraint=True):
        super().__init__(task)

        self._client = DashScopeClient(model_name=model_name,temperature=1.0)
        self._use_mean_constraint = use_mean_constraint

    @classmethod
    def update_running_mean(cls, new_score: float):
        """
        更新类变量 `_running_judge_mean`，用锁来保证线程安全。
        """
        with cls._update_lock:
            cls._running_judge_mean_fast = cls._alpha_fast * cls._running_judge_mean_fast + (1-cls._alpha_fast) * new_score
            cls._running_judge_mean_slow = cls._alpha_slow * cls._running_judge_mean_slow + (1-cls._alpha_slow) * new_score

    @classmethod
    def get_running_mean(cls):
        """
        获取当前的 `_running_judge_mean`。
        """
        with cls._update_lock:
            return cls._running_judge_mean_fast
    
    @classmethod
    def get_stable_mean(cls):
        with cls._update_lock:
            return cls._running_judge_mean_slow
    
    def pack_message(self, trajectory: Trajectory):
        """Pack trajectory into a message.
        
        Args:
            trajectory (Trajectory): trajectory to pack
        """
        messages = []
        
        assert len(trajectory.steps) >= 2 and trajectory.steps[1]['role'] == 'user', "trajectory must start with system message and then user message"
        task_query = trajectory.steps[1]['content']
        
        if self._use_mean_constraint:
            content=USER_PROMPT_WITH_MEAN_CONSTRAINT.format(
                task=task_query, 
                trajs=steps_to_msg(trajectory.steps[2:]),
                running_mean=self.get_running_mean(),
                mean_score=self.get_stable_mean(),
            )
        else:
            content=USER_PROMPT.format(
                task=task_query, 
                trajs=steps_to_msg(trajectory.steps[2:]),
            )
        
        messages.append(
            {
                "role": "user",
                "content": content
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
        """Calculate reward for a trajectory in specific environment.
        
        Args:
            trajectory (Trajectory): trajectory to calculate reward
            env (EnvClient): environment where the trajectory is executed
        """
        response = ""
        for chunk in self._client.chat_stream_with_retry(messages=self.pack_message(trajectory), max_retries=64):
            response += chunk
        
        # 默认分数
        score: float = 0.0

        if response:
            try:
                # 解析结果，兼容大小写与多余空格
                cf_match = re.search(r"Critical\s*Failure\s*:\s*(Yes|No)\b", response, re.IGNORECASE)
                intent_match = re.search(r"Intent\s*Comprehension\s*:\s*([01])\b", response, re.IGNORECASE)
                correct_match = re.search(r"Correct\s*Completion\s*:\s*([01])\b", response, re.IGNORECASE)

                critical = bool(cf_match and cf_match.group(1).strip().lower().startswith("y"))
                intent_score = int(intent_match.group(1)) if intent_match else 0
                correct_score = int(correct_match.group(1)) if correct_match else 0

                if critical:
                    score = 0.0
                else:
                    score = 0.2 * intent_score + 0.8 * correct_score
                    score = correct_score
            except Exception as e:
                logger.exception(f"Failed to parse LLM judge response: {e}. Raw response: {response!r}")
                score = 0.0
        else:
            logger.warning("Empty LLM judge response; setting score=0.0")
        
        self.update_running_mean(score)

        if not eject_llm_output:
            return score
        else:
            return score, response

@grader_manager.reg("llm-binary-no_constraint")
class LlmAsJudgeBinaryRewardCalculatorNoConstraint(LlmAsJudgeBinaryRewardCalculator):
    def __init__(self, task: Task, model_name='qwq-plus'):
        super().__init__(task, model_name, use_mean_constraint=False)