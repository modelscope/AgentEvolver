from typing import Any, cast
from agentevolver.client.env_client import EnvClient
from agentevolver.client.llm_client import DashScopeClient
from agentevolver.module.agent_flow.reward_calculator import GraderResult, RewardCalculator
from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory

from . import grader_manager

USER_PROMPT="""Based on the conversation trajectory above, evaluate the task completion quality using the framework provided.

Your evaluation should address the following dimensions in order:

**Step 1: Relevance Check (0 or proceed)**
- Are the solution steps relevant to the problem? If the approach is completely unrelated to the task requirements, assign 0 points immediately.
- If relevant, proceed to other evaluation dimensions.

**Step 2: Repetition Penalty Check**
- Does the agent get stuck in infinite loops or repeat identical steps endlessly?
- If there are infinite repetitions of the same steps, consider the relevance of existing steps:
 - If steps are relevant: Maximum 20 points
 - If steps are irrelevant: 0 points

**Step 3: Goal Achievement Assessment (Critical Binary Check)**
- Examine ALL steps comprehensively to determine if the task goal is truly achieved
- Do not be misled by superficial language - verify actual completion
- Check if there is a correct final answer or if the stated objective is genuinely accomplished

**MANDATORY SCORING CONSTRAINTS:**
- If steps are relevant AND goal is achieved/answer is correct: Score MUST be 60-100
- If steps are relevant BUT goal is not achieved/answer is incorrect: Score MUST be 0-40
- FORBIDDEN: Do not assign scores between 41-59

**Step 4: Additional Deductions (within the above constraints)**
- **Code Execution Errors**: Deduct points for runtime errors, bugs, or failed executions
- **Unnecessary/Irrelevant Steps**: Deduct points for redundant or off-topic actions

**Scoring Guidelines:**
- 90-100: Exceptional performance - goal achieved with efficient, clean execution
- 80-89: Strong performance - goal achieved with minor inefficiencies or small errors
- 70-79: Good performance - goal achieved with some unnecessary steps or code issues
- 60-69: Adequate performance - goal achieved but with notable problems
- 30-40: Poor performance - goal not achieved but relevant approach with some progress
- 10-29: Very poor performance - goal not achieved with major execution issues
- 1-9: Minimal relevant attempt - goal not achieved with severe problems
- 0: Complete failure - irrelevant approach or infinite repetition of irrelevant steps

**REMEMBER**: 
- No scores between 41-59 are allowed
- Goal achievement determines the 60+ vs 0-40 range
- Infinite repetition caps score at 20 (if steps are relevant) or 0 (if irrelevant)

Provide your detailed analysis first, explaining your reasoning for each evaluation dimension. Then assign a precise integer score following the mandatory constraints above.

First provide your detailed reasoning analysis, then output an integer score between 0-40 or 60-100 enclosed in <reward></reward> tags, e.g., <reward>75</reward>
"""


def steps_to_msg(steps: list[dict[str, Any]]) -> str:
    """
    Converts a list of step dictionaries into a formatted string representing the conversation trajectory.

    Args:
        steps (list[dict[str, Any]]): A list of dictionaries, where each dictionary represents a step in the conversation trajectory.

    Returns:
        str: A formatted string representing the conversation trajectory.
    """
    trajectory_text = ""
    assert steps[0]['role'] == 'assistant'  # ⭐ Ensures the first message is from the assistant
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
            raise ValueError("roles in trajectory must be assistant or user")  # ⭐ Raises an error if the role is neither 'assistant' nor 'user'
        trajectory_text += block.strip() + "\n\n"
    return trajectory_text

@grader_manager.reg("llm")
class LlmAsJudgeRewardCalculator(RewardCalculator):
    """
    A naive RewardCalculator that uses LLM as judge.
    """
    def __init__(self,task:Task, model_name='qwen3-235b-a22b-instruct-2507'):
        """
        Initializes the LlmAsJudgeRewardCalculator with a specific task and model name.

        Args:
            task (Task): The task to be evaluated.
            model_name (str, optional): The name of the language model to be used as the judge. Defaults to 'qwen3-235b-a22b-instruct-2507'.
        """
        super().__init__(task)
        self._client=DashScopeClient(model_name=model_name)  # ⭐ Initializes the LLM client with the specified model name

    def pack_message(self, trajectory: Trajectory):
        """Pack trajectory into a message.
        
        Args:
            trajectory (Trajectory): trajectory to pack
        """
        messages=[]
        
        assert len(trajectory.steps) >= 2 and trajectory.steps[1]['role'] == 'user', "trajectory must start with system message and then user message"
        query=trajectory.steps[1]['content']
        trajectory_text = f"Query: {query}\n"
        trajectory_text = "The following is the dialogue trace of the task execution:\n\n"
        trajectory_text+=steps_to_msg(trajectory.steps[2:])
        
        messages.append({"role": "user", "content": trajectory_text})
        messages.append({"role":"user","content":USER_PROMPT})
        return messages
    
    def calculate_reward(self, trajectory: Trajectory, env: EnvClient, instance_id: str) -> GraderResult:
        x,reason=cast(tuple[float,str],self._calculate_reward(trajectory,env,eject_llm_output=True))
        return {
            "score":x,
            "reason":reason
        }
        

    def _calculate_reward(self, trajectory: Trajectory, env:EnvClient, *, eject_llm_output:bool=False):
        """
        Calculates a reward for a given trajectory in a specific environment using an LLM.

        Args:
            trajectory (Trajectory): The trajectory for which the reward is calculated.
            env (EnvClient): The environment where the trajectory is executed.
            eject_llm_output (bool, optional): If True, the function returns both the score and the LLM's full response. Defaults to False.

        Returns:
            float or tuple: The normalized score, or a tuple of the score and the LLM's full response if `eject_llm_output` is True.
        """
        response=""
        for chunk in self._client.chat_stream_with_retry(messages=self.pack_message(trajectory),max_retries=64):
            response += chunk  # ⭐ Accumulate chunks of the LLM's response
        if response:
            import re
            reward_match = re.search(r'<reward>([\d\.]+)</reward>', response.strip())
            if reward_match:
                score = float(reward_match.group(1))
                score = max(0.0, min(100.0, score))/100.0  # ⭐ Normalize the extracted score
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