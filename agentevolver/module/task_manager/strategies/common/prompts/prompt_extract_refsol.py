# this prompt is used to summarize the agent's trajectory only
import json
from typing import Optional, Sequence, Tuple

from agentevolver.module.task_manager.env_profiles import EnvProfile
from agentevolver.schema.task import Task, TaskObjective
from agentevolver.schema.trajectory import Trajectory
from .prompt_summarize import _get_action_observation_pair


AGENT_SUMMARIZE_SYSTEM_PROMPT = """# ROLE
You are a **Real-World Task Solution Expert**.  
Your job is to analyze an agent's API interaction history and extract a solution from it.

---

# OBJECTIVES
- Analyze the recorded API calls to identify the actual functional capabilities demonstrated.
- Provide the sequence of technical steps that directly accomplishes the task.
- The solution should be self-contained, i.e., it should include all necessary API calls and function definitions.

# OUTPUT FORMAT
For each identified task, output exactly one block in this format:

<task>
{
  "confidence": [0.0 - 1.0],
  "action_sequence": "[Technical sequence that directly solves the task]"
}
</task>

---

# GOOD EXAMPLES
<task>
{
  "confidence": 1.0,
  "action_sequence": "# step0\nbalance = apis.venmo.get_balance()\nif balance >= 150:\n    print('Yes')\nelse:\n    print('No')"
}
</task>

<task>
{
  "confidence": 1.0,
  "action_sequence": "# step0\n[click('https://www.taobao.com')]\n# step1\n[search('red women heels price<100 delivery:2025-08-22')]"
}
</task>
"""


def get_task_summarize_prompt(
    trajectories: Sequence[Trajectory],
    old_objectives: Sequence[TaskObjective],
    profile: EnvProfile | None,
) -> tuple[str, str]:
    x = ""
    idx = 0
    for traj in trajectories:
        pairs = _get_action_observation_pair(traj)
        x += f"## Record {idx}\n"
        x += f"### History\n"
        for step_idx, history in enumerate(pairs):
            x+=f""">>> STEP {step_idx} <<<
<|ACTION|>
{history[0]}
<|END|>

<|OBSERVATION|>
{history[1]}
<|END|>
"""
            pass
        assert traj.reward is not None
        x += f"### Reward: {traj.reward.outcome}\n{traj.reward.description}\n"
        idx += 1

    objectives: list[str] = [x.objective for x in old_objectives if x.objective is not None]

    user_prompt = f"""Please analyze the following agent interaction sequence and abstract solution from it:

{x}

# Now Start

Please identify the specific tasks the agent is attempting to complete in these interactions, and abstract solution following the specified format.
"""

    return AGENT_SUMMARIZE_SYSTEM_PROMPT, user_prompt


def parse_tasks_from_response(response: str) -> str|None:

    gts: list[str] = []
    try:
        import re

        task_matches = re.findall(r"<task>(.*?)</task>", response, re.DOTALL)

        for task_content in task_matches:
            t = json.loads(task_content)

            if (
                "confidence" not in t
                or "action_sequence" not in t
            ):
                continue
            ground_truth=t["action_sequence"]
            gts.append(ground_truth)

    except Exception as e:
        print(f"Error parsing tasks: {e}")
    
    if len(gts)!=0:
        return gts[0]
    else:
        return None