import torch
from agentevolver.module.adv_processor.candidate_prompt import sys_msg_1003, THRESHOLD


def rescale_score(score: float, threshold: float = THRESHOLD) -> float:
    """
    Rescales the input score based on the specified threshold.
    If the threshold is 0.5, the score is returned as is.
    If the threshold is 0, the score is rescaled to range from -1 to 1.

    Args:
        score (float): The input score to be rescaled.
        threshold (float): The threshold value for rescaling. Default is 0.

    Returns:
        float: The rescaled score.
    """
    if threshold == 0.5:
        return score
    elif threshold == 0:
        return (score - 0.5) * 2
    else:
        raise ValueError("Threshold must be either 0 or 0.5.")

def get_positive_mask(scores: torch.Tensor | float, threshold: float = THRESHOLD) -> torch.Tensor | bool:
    """
    Determines whether scores are positive based on the given threshold.
    This function can handle a single float or an entire PyTorch tensor.
    Used in:
    1. Generating evaluation prompts
    2. Determining which are positive trajectories and which are negative trajectories in evaluation statistics
    Args:
        scores: A single score or a batch of scores.
        threshold: The threshold for determining positivity.

    Returns:
        A boolean value or boolean tensor indicating whether the score is positive.
    """
    return scores > threshold

def build_batch_adv_evaluation_prompt(
        query: str,
        steps: list[dict],
        overall_adv: float,
        max_step_chars: int = 2000,
) -> list[dict]:
    """
    Constructs a prompt for evaluating a series of steps in a task. The prompt includes a system message with detailed evaluation rules and a user message with the task description, solution trajectory, and overall performance score.

    Args:
        query (str): The original task description.
        steps (list[dict]): A list of dictionaries, each representing a step in the solution trajectory.
        overall_adv (float): The overall performance score, indicating the quality of the final answer.
        max_step_chars (int, optional): The maximum number of characters allowed for each step. Defaults to 2000.

    Returns:
        list[dict]: A list of dictionaries, each containing the system message and the user message for the evaluation prompt.
    """
    # polarity = "positive" if overall_adv > 0.5 else "negative"
    is_pos = get_positive_mask(overall_adv)
    polarity = "positive" if is_pos else "negative"
    
    
    sys_msg = f"""You are an expert *process reward evaluator*, specializing in **attributional analysis** of multi-step solution trajectories.

**INPUT STRUCTURE:** The single message you receive always contains three labelled sections:
  1.  **TASK DESCRIPTION**   – The user's original request.
  2.  **SOLUTION TRAJECTORY** – A strictly numbered list of assistant steps. Each step describes an `ACTION` taken (and optionally an `OBSERVATION`).
  3.  **OVERALL PERFORMANCE SCORE** – A scalar value (integer or float) summarising the final answer quality relative to the task. **>0** indicates the overall outcome was **advantageous** (successful/helpful). **<0** indicates the overall outcome was **disadvantageous** (unsuccessful/unhelpful).

**YOUR TASK (ATTRIBUTIONAL ANALYSIS):** Analyze the `SOLUTION TRAJECTORY` and **attribute the contribution of each numbered step** to the final `OVERALL PERFORMANCE SCORE`. 

**EVALUATION RULES (By Score Sign):**

*   **If OVERALL PERFORMANCE SCORE is POSITIVE (> {THRESHOLD:+.1f}):**
    *   An individual step is classified as **GOOD** if its `ACTION` (and its result, if `OBSERVATION` is present) **contributed positively** to achieving the final advantageous outcome. This includes:
        *   Making a significant **incremental improvement** towards the solution.
        *   **Correctly executing** a necessary sub-task.
        *   **Preserving or building upon** correct prior steps.
    *   An individual step is classified as **BAD** if its `ACTION` (or result) was **neutral, irrelevant, or detrimental** to the eventual positive outcome.

*   **If OVERALL PERFORMANCE SCORE is NEGATIVE (≤ {THRESHOLD:+.1f}):**
    *   An individual step is classified as **GOOD** **only** if its `ACTION` (and its result, if `OBSERVATION` is present) **actively attempted to mitigate or correct** an existing problem or error trajectory. Specifically:
        *   **Successfully fixing** an earlier error.
        *   **Actively moving the solution back towards correctness** after a misstep.
        *   **Preventing a further degradation** of the situation.
    *   An individual step is classified as **BAD** if its `ACTION` (or result) was **neutral, irrelevant, introduced a new error, or failed to correct an existing error**, thereby contributing to or failing to improve the eventual negative outcome.

**FOCUS:** Ignore superficial elements (politeness, formatting). Evaluate **strictly** based on the **technical impact and causal contribution** of the step's `ACTION` (and `OBSERVATION` if present) on the final outcome, relative to the `TASK DESCRIPTION`.

**OUTPUT FORMAT:** Reply IN THE REQUIRED OUTPUT FORMAT and output nothing else.

"""
    def _trim(s: str) -> str:
        if not s: return ""
        return s if len(s) <= max_step_chars else s[:max_step_chars] + "\n…"

    user_parts = [
        "### TASK DESCRIPTION",
        query,
        "",
        f"### SOLUTION TRAJECTORY  (total {len(steps)} steps)",
    ]

    for i, st in enumerate(steps):
        block = [
            f">>> EVAL-STEP {i} <<<",
            "<|ACTION|>",
            _trim(st.get("action","")),
            "<|END|>",
        ]
        obs = st.get("observation")
        if obs:
            block += ["<|OBSERVATION|>", _trim(obs), "<|END|>"]
        user_parts.append("\n".join(block))

    user_parts += [
        "",
        "---",
        f"**OVERALL PERFORMANCE SCORE {overall_adv:+.4f} ({polarity})**",
        "Evaluation reminder:",
        "• Positive SCORE → Did this step IMPROVE the answer?",
        "• Negative SCORE → DIAGNOSIS + FIX + EVIDENCE (quoted). If evidence missing → BAD.",
        "  (Continuing wrong plan / repeating same failure / finalising wrong result → BAD)",
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
        {"role": "user", "content": "\n".join(user_parts)},
    ]


def build_batch_reward_evaluation_prompt(
        query: str,
        steps: list[dict],
        overall_adv: float,
        max_step_chars: int = 2000,
) -> list[dict]:
    """
    Constructs a structured prompt for evaluating the reward of a batch of steps in a given task.

    Args:
        query (str): The original request or task description.
        steps (list[dict]): A list of dictionaries, each representing a step in the solution trajectory.
        overall_adv (float): The overall reward score indicating the success or failure of the task.
        max_step_chars (int, optional): The maximum number of characters allowed for each step. Defaults to 2000.

    Returns:
        list[dict]: A list of dictionaries, each containing a 'role' and 'content' key, representing the system and user messages.
    """
    # polarity = "positive" if overall_adv > 0.5 else "negative"
    is_pos = get_positive_mask(overall_adv)
    polarity = "positive" if is_pos else "negative"
    

    sys_msg = sys_msg_1003
    def _trim(s: str) -> str:
        if not s: return ""
        return s if len(s) <= max_step_chars else s[:max_step_chars] + "\n…"

    user_parts = [
        "### TASK DESCRIPTION",
        query,
        "",
        f"### SOLUTION TRAJECTORY  (total {len(steps)} steps)",
    ]

    for i, st in enumerate(steps):
        block = [
            f">>> EVAL-STEP {i} <<<",
            "<|ACTION|>",
            _trim(st.get("action","")),
            "<|END|>",
        ]
        obs = st.get("observation")
        if obs:
            block += ["<|OBSERVATION|>", _trim(obs), "<|END|>"]
        user_parts.append("\n".join(block))

    user_parts += [
        "",
        "---",
        f"**OVERALL PERFORMANCE SCORE {overall_adv:+.4f} ({polarity})**",
        "Evaluation reminder:",
        "• Positive SCORE → Did this step IMPROVE the answer?",
        "• Negative SCORE → DIAGNOSIS + FIX + EVIDENCE (quoted). If evidence missing → BAD.",
        "  (Continuing wrong plan / repeating same failure / finalising wrong result → BAD)",
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
        {"role": "user", "content": "\n".join(user_parts)},
    ]