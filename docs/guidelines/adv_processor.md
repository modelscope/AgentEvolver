## Overview

The **Advantage Processor** is one of core components of the AgentEvolver framework, responsible for implementing the **`self-attributing`** mechanism. Traditional reinforcement learning methods face the **credit assignment problem** in long-horizon tasks: they cannot distinguish which actions in a trajectory are valuable versus unproductive, leading to inefficient learning. The Self-Attributing mechanism addresses this by leveraging LLM's causal reasoning capabilities to decompose learning signals into **process quality** (logical correctness of actions) and **outcome effectiveness** (final success), enabling precise step-wise credit assignment.

### **Core Implementation: ADCA-GRPO**

The primary implementation of the Advantage Processor is **ADCA-GRPO**. ADCA-GRPO uses a powerful LLM to perform causal analysis on trajectories, assigning an **attribution signal (GOOD/BAD)** to each step. This signal is then combined with the traditional outcome signal to produce a fine-grained advantage, leading to significantly improved sample efficiency. Experimental results demonstrate that ADCA-GRPO achieves approximately 10% performance improvement and 40% reduction in training steps compared to traditional GRPO methods.

The modular design allows for future extensions with alternative credit assignment paradigms.

-----

## ADCA-GRPO: Implementation Workflow

The ADCA-GRPO advantage calculation is an end-to-end pipeline that transforms raw trajectory data into a fine-grained advantage signal. Think of it as teaching the agent to distinguish "good reasoning steps" from "bad ones" - similar to how a teacher evaluates both the problem-solving process and the final answer.

### **Stage 1: Semantic Evaluation - "Was this step done correctly?"**

**Why we need this:** Traditional RL only knows if the final outcome was good or bad, but can't tell which intermediate steps were helpful. We need an LLM to act as a "step-by-step evaluator."

**Example:** In an AppWorld task, the LLM might evaluate "calling the wrong API" as BAD, while "correctly parsing parameters" gets marked as GOOD.

**How it works:** The system uses an LLM to generate step-wise `GOOD`/`BAD` labels for each trajectory in the batch.

  * **Key Function**: `evaluate_step_flags_parallel_sync` (from `semantic_attribution.py`)

```python
# From: semantic_attribution.py

async def evaluate_step_flags_parallel(
    tokenizer,
    batch,
    overall_score_source: str = "advantages",
    model_name: str = "qwen-max",
    # ... other parameters
) -> Tuple[List[List[bool]], Dict]:
    ...
```

This function orchestrates the evaluation process:

1.  **Preparing Data**: Decodes prompts and responses for each sample in the `batch`.
2.  **Constructing Prompts**: Builds detailed prompts for the LLM including the query, full rollout, and final outcome score.
3.  **Parallel API Calls**: Manages asynchronous, parallel API calls with rate limiting and retries via `_async_safe_query`.
4.  **Parsing Results**: Uses `parse_batch_evaluation_result` to convert LLM's natural language response into structured boolean flags (`True` for `GOOD`, `False` for `BAD`).

**Output:** `step_flags` - a list of lists containing boolean labels for each step in every trajectory.

### **Stage 2: Signal Fusion - "How do we balance process vs. outcome?"**

**Why we need this:** Just like evaluating a student's work, we need to consider both the problem-solving steps (process) AND the final answer (outcome). The key insight is to normalize these signals independently to prevent one from overwhelming the other.

**How it works:** With the `step_flags`, the pipeline calculates the final advantage signal by fusing attribution and outcome signals.

  * **Key Function**: `compute_prm_grpo_advantages` (from `adca_grpo.py`)

```python
# From: adca_grpo.py

def compute_prm_grpo_advantages(
    batch,
    step_flags: List[List[bool]],
    hyper: Optional[PRMHyper] = None,
    scheme: str = "decouple",
) -> dict:
    ...
```

For the recommended `"decouple"` scheme, this calls `_build_decouple`, which implements the core logic:

  * **Utility Function**: `_build_decouple` (from `adca_grpo.py`)

```python
# From: adca_grpo.py

def _build_decouple(
    orm_full_scores: torch.Tensor,
    step_flags: List[List[bool]],
    step_ids: torch.Tensor,
    group_ids: torch.Tensor,
    hyper: "PRMHyper"
) -> Tuple[List[List[float]], Dict]:
    ...
```

**The decouple process:**

1.  **Construct Raw PRM Rewards**: Maps `GOOD`/`BAD` flags to numerical scores (e.g., `+/- hyper.fix_base`), creating "process quality" rewards.
2.  **Independent Z-Score Normalization**: The crucial step that prevents signal interference:
      * Step-level PRM rewards are normalized using `_group_zscore_on_steps` → `prm_rewards_std`
      * Trajectory-level outcome scores are independently normalized → `orm_scores_std`
3.  **Fuse Rewards**: Combines the standardized signals with `hyper.alpha` controlling the balance between attribution (process) and outcome signals.

**Output:** `step_rewards` - fused, step-level reward values for each trajectory.

### **Stage 3: Advantage Computation - "How do we guide policy learning?"**

**Why we need this:** The step-level rewards must be converted to token-level advantages that the PPO/GRPO optimizer can use. This involves computing future reward expectations and mapping them to individual tokens.

**How it works:** Converts step-level rewards into token-level advantage tensors.

  * **Key Function**: `suffix_sum_on_steps` (from `adca_grpo.py`)

Calculates advantage for each step by computing cumulative sum of future rewards (suffix sum). This tells each step "how much future reward you can expect from this point."

  * **Key Function**: `broadcast_step_adv_to_tokens` (from `adca_grpo.py`)

Maps step-level advantages to token level using the `step_ids` tensor, which tracks which step each token belongs to. All tokens in a step receive that step's advantage value.

**Final Output:** The `advantages` tensor is written back into the `batch` object for policy training.

-----

## Key Parameters & Configuration ⚙️

Here is a comprehensive list of parameters based on the source code and configuration structure.

### **1. Global & General Settings**

**`enable`** (*bool*)
: **Master switch for the module**. If `true`, the attribution and advantage rewriting logic will be executed.

**`enable_adca_metric`** (*bool*)
: Enables additional ADCA monitoring metrics. Recommended to turn on for debugging and analysis.

### **2. LLM Attribution Service**

These parameters control the behavior of calling the large language model for `GOOD`/`BAD` labeling.

**`evaluation_type`** (*str*)
: The evaluation method. Currently fixed to `"api"`, indicating LLM calls via an API.

**`model`** (*str*)
: The **LLM model name** used for semantic evaluation, e.g., `"qwen-plus"`.

**`concurrent`** (*int*)
: The number of **parallel API requests**. Adjust based on your API rate limits; a value between 5-20 is recommended.

**`api_max_retries`** (*int*)
: The **maximum number of retries** for a failed API request. The default is 200, which is very robust.

**`llm_evaluation_log_dir`** (*str*)
: (Optional) The directory path to save LLM request/response logs. Highly recommended for debugging.

**`skip_type`** (*str*)
: **Strategy for skipping LLM calls** to save costs. `"skip_small_adv"` (recommended) skips samples with near-zero advantage. `"skip_all_neg"` skips negative-reward samples. `"none"` disables skipping.

### **3. ADCA-GRPO Core Algorithm (`adca_grpo` submodule)**

These parameters directly influence the reward construction, fusion, and advantage computation.

**`prm_scheme`** (*str*)
: The reward fusion scheme. **`"decouple"` is strongly recommended** as it normalizes attribution and outcome rewards separately, making it more stable. `"allocation"` is an alternative experimental scheme.

**`do_batch_norm`** (*bool*)  
: Whether to perform **group-wise Z-Score normalization** on step rewards. **Strongly recommended to be `true`** as it is key to training stability.

**`equal_trajectory_weight`** (*bool*)  
: Whether to treat each trajectory equally during normalization (GRPO). Recommended to be `true`. If `false`, all steps are pooled together (GSPO), which can be useful in noisy environments.

**`fix_base`** (*float*)  
: (For `decouple` scheme) The **base numerical value** to map `GOOD`/`BAD` labels to. For example, 0.2 means `GOOD`=+0.2 and `BAD`=-0.2.

**`alpha`** (*float*)  
: The **weighting coefficient for the attribution reward (α)**. This is one of the most important hyperparameters for balancing process quality vs. final outcome.

**`beta`** (*float*)  
: The **weighting coefficient for the outcome reward (β)**. For simplicity, this is often fixed to 1.0 during experiments.

**`orm_distribution`** (*str*)  
: The **distribution method for the outcome reward (ORM)**. `"last_step"` (recommended) applies it only to the final step, while `"all_steps"` distributes it across all steps.

**`prm_steps`** (*int*)  
: **Enables attribution advantage for the first N epochs only**. This is an effective strategy for cost control, allowing the agent to rely on its learned policy in later training stages.

**`enable_length_normalization`** (*bool*)  
: (For `decouple` scheme only) Whether to enable trajectory length normalization (dividing rewards by sqrt of step count). May help balance contributions from long vs. short trajectories. Default is `false`.

-----

## Quick Start & Recommended Configuration 🚀

### **Step 1: Set Environment Variable**

Before starting, ensure your API key is set in your terminal environment:

```bash
export DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
```

### **Step 2: Recommended Configuration**

Here is a more complete and ready-to-use baseline configuration. It uses the stable `decouple` scheme and includes detailed settings for cost optimization and monitoring.

```yaml
# Semantic evaluation & ADCA-GRPO
attribution_driven_credit_assignment:
  # 1. Master switch
  enable: true

  # 2. LLM Attribution Service settings
  evaluation_type: "api"
  model: "qwen-max"                # Recommend using a powerful model
  concurrent: 10                    # Adjust based on your API QPS limits
  api_max_retries: 200
  llm_evaluation_log_dir: "/path/to/your/logs" # Strongly recommended for debugging

  # 3. ADCA-GRPO Core Algorithm settings
  adca_grpo:
    # Must be "decouple" for the most stable and validated scheme
    prm_scheme: "decouple"
    
    # Must be true to ensure signal scale consistency and training stability
    do_batch_norm: true
    
    # Generally recommended to be true to ensure equal contribution from each trajectory
    equal_trajectory_weight: true
    
    # Base value for GOOD/BAD labels under the "decouple" scheme
    fix_base: 0.2
    
    # Key hyperparameter to tune; start with a small value (e.g., 0.05 ~ 0.2)
    alpha: 0.1 # This corresponds to 'α' in the paper
    
    # Beta is implicitly 1.0 in the code and fixed in experiments
    # beta: 1.0
    
    # Recommended to be "last_step" as it is more intuitive
    orm_distribution: "last_step"
    
    # Balance effectiveness and API cost; enables attribution for the first 20 epochs
    prm_steps: 20
    
    # Recommended to save API costs by not evaluating low-value trajectories
    skip_type: "skip_small_adv"
    
    # Recommended to be true to monitor the attribution module's internal state
    enable_adca_metric: true
    
    # Default is false; can be enabled for tasks with high variance in trajectory length
    enable_length_normalization: false
```