import json
from typing import Optional, Sequence, Tuple

from beyondagent.schema.task import Task, TaskObjective
from beyondagent.schema.trajectory import Trajectory


AGENT_SUMMARIZE_SYSTEM_PROMPT = """
You are a *Task Abstraction Expert*. Your specialty is to inspect an agent's interaction history and distill concrete, goal-oriented tasks from it.

========================  YOUR JOB  ========================
1. Inspect the interactions.
2. Identify the specific goal or task the agent is attempting to achieve.
3. Abstract each goal into a clear, concise **task description**, a **query** (suitable for search or training), and the **minimal action sequence** that successfully completes the task.

=====================  ABSTRACTION RULES  ==================
- Focus on clear, goal-directed behaviour; ignore purely random exploration.  
- Group similar behaviour patterns into the same task.  
- Every task must have **at least one** action sequence that was executed successfully.  
- Each task needs an explicit completion criterion.  
- All actions listed in an action sequence must be valid and directly executable by the agent.
- All actions listed in an action sequence must be included in the available APIs of the current environment state.
- Ensure that all actions listed in an action sequence are combined into a minimum sequence from the initial state of the environment to the completion of the task. No additional information or skipped steps are allowed.

========================  OUTPUT FORMAT  ===================
For every task you identify, output exactly one block in the form below:

<task>
{
  "query": "[A succinct search / training query—results only, no extra guidance.]",
  "confidence": "[0.0 - 1.0, your confidence in this abstraction]",
  "action_sequence": "[A minimal sequence]"
}
</task>

===========================  EXAMPLE  ======================
EXAMPLE 1
<task>
{
  "query": "I want to get my password for supervisor account",
  "confidence": 1.0,
  "action_sequence": "# step0\nprint(apis.api_docs.show_app_descriptions())\n# step1\nprint(apis.api_docs.show_api_descriptions(app_name='supervisor'))\n# step2\nprint(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))\n# step3\nprint(apis.supervisor.show_account_passwords())\npasswords = apis.supervisor.show_account_passwords()"
}
</task>

EXAMPLE 2
<task>
{
  "query": "Show the files under /home/admin",
  "confidence": 1.0,
  "action_sequence": "# step0\ncd /home/admin\n # step1\nls ."
}
</task>

EXAMPLE 3
<task>
{
  "query": "Find me a red shoes in this website",
  "confidence": 1.0,
  "action_sequence": "# step0\n[click('https://www.taobao.com')]\n # step1\n[search('red shoes')]"
}
</task>
"""


def _get_action_observation_pair(traj: Trajectory) -> list[tuple[str, str]]:
    res = []
    for idx, step in enumerate(traj.steps):
        assert "role" in step, "steps must have role field"
        if step["role"] == "assistant" and idx + 1 < len(traj.steps):
            next_step = traj.steps[idx + 1]
            # As there is no standard for environments, we do not know whether it will response as user or tool.
            if next_step["role"] == "tool":
                # get observation from tool message
                observation = next_step["content"]
            elif next_step["role"] == "user":
                # get observation from user message
                observation = next_step["content"]
            else:
                continue
            res.append((step["content"], observation))

    return res


def get_task_summarize_prompt(
    trajectories: Sequence[Trajectory],
    old_objectives: Sequence[TaskObjective],
    len_history: int = 2,
) -> tuple[str, str]:
    """获取任务摘要 prompt"""
    x = ""
    idx = 0
    for traj in trajectories:
        pairs = _get_action_observation_pair(traj)
        for k, v in enumerate(pairs):
            histories = pairs[max(0, k - len_history) : k]
            x += f"## Record {idx}\n"
            x += f"### History\n"
            for history in histories:
                x += f"assistant:\n{history[0]}\n->\nobservation:{history[1]}\n\n"
            x += f"### Latest Action\n{v[0]}\n"
            x += f"### Latest Observation\n{v[1]}\n"
            x += f"### Reward: {traj.reward.outcome}\n{traj.reward.description}\n"
            idx += 1

    objectives: list[str] = []
    for ob in old_objectives:
        if isinstance(ob.objective, str):
            objectives.append(ob.objective)

    user_prompt = f"""Please analyze the following agent interaction sequence and abstract specific tasks from it:

{x}

# Old Objectives
You have already explored the following objectives:

{objectives}

Please avoid repeating the objectives in the current exploration.

# Now Start

Please identify the specific tasks the agent is attempting to complete in these interactions, and abstract them into clear task descriptions and queries following the specified format.
"""

    return AGENT_SUMMARIZE_SYSTEM_PROMPT, user_prompt


def parse_tasks_from_response(task: Task, response: str) -> list[TaskObjective]:
    """从响应中解析任务列表"""
    task = task.copy()

    tasks: list[TaskObjective] = []
    try:
        import re

        # 找到所有<task>标签中的内容
        task_matches = re.findall(r"<task>(.*?)</task>", response, re.DOTALL)

        for task_content in task_matches:
            t = json.loads(task_content)

            # 检查必要字段
            if (
                "query" not in t
                or "confidence" not in t
                or "action_sequence" not in t
            ):
                continue
            task.query = t["query"]
            tasks.append(
                TaskObjective(
                    task=task,
                    confidence=t["confidence"],
                    ground_truth=t["action_sequence"],
                    reward=None,
                )
            )

    except Exception as e:
        print(f"Error parsing tasks: {e}")

    return tasks
