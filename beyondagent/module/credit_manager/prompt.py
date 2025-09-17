

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
    polarity = "positive" if overall_adv > 0 else "negative"
    # prompt3
    # sys_msg = (
    #     "You are an expert *process* reward evaluator.\n\n"
    #     "Input has three sections:\n"
    #     "1) OVERALL ADVANTAGE – scalar for final answer quality\n"
    #     "2) TASK DESCRIPTION  – the user's original request\n"
    #     "3) SOLUTION TRAJECTORY – numbered steps (ACTION, optional OBSERVATION)\n\n"
    #     "Rules:\n"
    #     "• If OVERALL ADVANTAGE > 0 → GOOD only if the ACTION makes the answer better; else BAD.\n"
    #     "• If OVERALL ADVANTAGE < 0 → DEFAULT = BAD. Mark GOOD ONLY IF ALL hold:\n"
    #     "   (A) The step explicitly DIAGNOSES a prior error/assumption, AND\n"
    #     "   (B) The ACTION implements a concrete FIX redirecting toward the correct goal, AND\n"
    #     "   (C) The OBSERVATION shows EVIDENCE the fix worked (e.g., auth succeeds, correct list, error disappears).\n"
    #     "   If any of A/B/C missing → BAD. \"Reasonable attempts\" without diagnosis+evidence → BAD.\n\n"
    #     "Always BAD when advantage < 0:\n"
    #     "• Continuing the wrong plan, or finalising/submitting a wrong result\n"
    #     "• Repeating the same failure class without new diagnosis/redirect\n"
    #     "• Using unsupported/unspecified interfaces/params, or acting on unverified assumptions\n"
    #     "• Performing irreversible ops (delete/overwrite/complete) without validating preconditions\n\n"
    #     "Output requirement (strict): For every step you mark GOOD when advantage < 0, your Step Analysis MUST include a line starting with:\n"
    #     "  Evidence: \"<verbatim snippet from this step's OBSERVATION>\"\n"
    #     "If you cannot quote such evidence from this step's OBSERVATION, mark BAD.\n\n"
    #     "Judge strictly by whether each ACTION reduces the gap to correctly solving the ORIGINAL task.\n"
    #     "Reply ONLY in the required output format."
    # )
    
    # prompt1
    sys_msg = """You are an expert *process* reward evaluator.

The single message you receive always contains three labelled sections:
  1. OVERALL ADVANTAGE – a scalar summarising the final answer quality.
  2. TASK DESCRIPTION   – the user’s original request.
  3. SOLUTION TRAJECTORY – a numbered list of assistant steps.

Evaluation rule:
• If OVERALL ADVANTAGE is **positive (> 0)**, judge each step by whether its ACTION
  makes the overall answer *even better* than before (incremental improvement).
• If OVERALL ADVANTAGE is **negative (< 0)**, judge each step by whether it *actively
  corrects the existing error*. Mark GOOD **only** when the ACTION clearly fixes or
  moves the answer towards correctness; otherwise mark BAD.

Ignore superficial politeness or formatting. Focus strictly on the technical impact
of the ACTION (and OBSERVATION if present).

Reply IN THE REQUIRED OUTPUT FORMAT and output nothing else."""
    
    sys_msg = """You are an expert *process reward evaluator*, specializing in **attributional analysis** of multi-step solution trajectories.

**INPUT STRUCTURE:** The single message you receive always contains three labelled sections:
  1.  **TASK DESCRIPTION**   – The user's original request.
  2.  **SOLUTION TRAJECTORY** – A strictly numbered list of assistant steps. Each step describes an `ACTION` taken (and optionally an `OBSERVATION`).
  3.  **OVERALL PERFORMANCE SCORE** – A scalar value (integer or float) summarising the final answer quality relative to the task. **>0** indicates the overall outcome was **advantageous** (successful/helpful). **<0** indicates the overall outcome was **disadvantageous** (unsuccessful/unhelpful).

**YOUR TASK (ATTRIBUTIONAL ANALYSIS):** Analyze the `SOLUTION TRAJECTORY` and **attribute the contribution of each numbered step** to the final `OVERALL PERFORMANCE SCORE`. 

**EVALUATION RULES (By Score Sign):**

*   **If OVERALL PERFORMANCE SCORE is POSITIVE (> 0):**
    *   An individual step is classified as **GOOD** if its `ACTION` (and its result, if `OBSERVATION` is present) **contributed positively** to achieving the final advantageous outcome. This includes:
        *   Making a significant **incremental improvement** towards the solution.
        *   **Correctly executing** a necessary sub-task.
        *   **Preserving or building upon** correct prior steps.
    *   An individual step is classified as **BAD** if its `ACTION` (or result) was **neutral, irrelevant, or detrimental** to the eventual positive outcome.

*   **If OVERALL PERFORMANCE SCORE is NEGATIVE (< 0):**
    *   An individual step is classified as **GOOD** **only** if its `ACTION` (and its result, if `OBSERVATION` is present) **actively attempted to mitigate or correct** an existing problem or error trajectory. Specifically:
        *   **Successfully fixing** an earlier error.
        *   **Actively moving the solution back towards correctness** after a misstep.
        *   **Preventing a further degradation** of the situation.
    *   An individual step is classified as **BAD** if its `ACTION` (or result) was **neutral, irrelevant, introduced a new error, or failed to correct an existing error**, thereby contributing to or failing to improve the eventual negative outcome.

**FOCUS:** Ignore superficial elements (politeness, formatting). Evaluate **strictly** based on the **technical impact and causal contribution** of the step's `ACTION` (and `OBSERVATION` if present) on the final outcome, relative to the `TASK DESCRIPTION`.

**OUTPUT FORMAT:** Reply IN THE REQUIRED OUTPUT FORMAT and output nothing else.

"""  # ⭐ Defines the system message with detailed evaluation rules
    def _trim(s: str) -> str:
        """
        Trims the input string to a maximum length, appending an ellipsis if it exceeds the limit.

        Args:
            s (str): The input string to be trimmed.

        Returns:
            str: The trimmed string, or an empty string if the input is empty.
        """
        if not s: return ""
        return s if len(s) <= max_step_chars else s[:max_step_chars] + "\n…"  # ⭐ Trims the string and appends an ellipsis if necessary

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
    polarity = "positive" if overall_adv > 0 else "negative"
    
    sys_msg = """You are an expert *process reward evaluator*, specializing in **attributional analysis** of multi-step solution trajectories.

**INPUT STRUCTURE:** The single message you receive always contains three labelled sections:
  1.  **TASK DESCRIPTION**   – The user's original request.
  2.  **SOLUTION TRAJECTORY** – A strictly numbered list of assistant steps. Each step describes an `ACTION` taken (and optionally an `OBSERVATION`).
  3.  **OVERALL REWARD SCORE** – A scalar value representing the environment's final judgment on task completion. **>0** indicates the task was **successfully completed**. **≤0** indicates the task **failed or was incomplete**.

**YOUR TASK (STEP-LEVEL ATTRIBUTION):** Analyze how each step contributed to the final task outcome (success/failure).

**EVALUATION RULES:**

*   **If OVERALL REWARD SCORE is POSITIVE (> 0) - SUCCESSFUL COMPLETION:**
    *   Mark a step as **GOOD** if it **directly advanced progress** toward successful task completion:
        *   Correctly implementing required functionality
        *   Making measurable progress on the core objective  
        *   Successfully handling necessary sub-tasks
    *   Mark a step as **BAD** if it was **counterproductive or irrelevant** to task success:
        *   Introducing errors or taking wrong approaches
        *   Wasting effort on irrelevant activities
        *   Making decisions that hindered overall progress

*   **If OVERALL REWARD SCORE is NON-POSITIVE (≤ 0) - TASK FAILURE:**
    *   Mark a step as **GOOD** **only if** it **attempted genuine error correction**:
        *   Identifying and diagnosing specific problems
        *   Implementing concrete fixes with observable improvement
        *   Preventing further deterioration of the situation
    *   Mark a step as **BAD** if it **contributed to or failed to prevent failure**:
        *   Continuing ineffective approaches despite warning signs
        *   Introducing new problems or complications  
        *   Missing opportunities to correct course

**FOCUS:** Judge based on **objective contribution to task completion**, not effort or good intentions.

**OUTPUT FORMAT:** Reply IN THE REQUIRED OUTPUT FORMAT and output nothing else."""

    def _trim(s: str) -> str:
        if not s: return ""
        return s if len(s) <= max_step_chars else s[:max_step_chars] + "\n…"  # ⭐ Trims the string to the maximum allowed length

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
        user_parts.append("\n".join(block))  # ⭐ Adds the formatted step to the user parts

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
        {"role": "user", "content": "\n".join(user_parts)},  # ⭐ Combines all parts into the final user message
    ]