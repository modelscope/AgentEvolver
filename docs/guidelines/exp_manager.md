## 🚀 Overview
Exploration in complex environments is a key challenge for autonomous agents. Traditional reinforcement learning approaches rely heavily on trial-and-error, often generating redundant trajectories and converging slowly. In contrast, humans efficiently leverage past experiences to guide future actions, learning faster and more systematically.

The **`Self-Navigating`** framework adopts this principle by enabling agents to internalize and reuse prior experiences. It shifts exploration from unguided trial-and-error to structured, knowledge-driven self-improvement, improving both learning efficiency and policy quality.

At the core of the framework, the **Experience Manager** oversees all aspects of experience handling, including:

1. **Experience Pool Management** – Constructing and updating the experience pool with new trajectories and summaries.
2. **Experience Mode Control** – Determining whether to add experiences during rollouts and whether to remove experience information during training.
3. **Rollout & Training Context Management** – Providing relevant historical context during rollouts and maintaining experience-strippped training messages.
4. **Training Loss Management** – Aggregating and processing losses with respect to experience-based adjustments for stable learning.

## 🧩 Core Features
The `Self-Navigating` framework enhances reinforcement learning by transforming how agents **create, reuse, and refine experiences**.
It introduces structured mechanisms that make exploration **more efficient**, **context-aware**, and **self-evolving**.

At its core are two classes:

- **`ExperienceManager`** — handles experience scheduling and allocation.  
- **`ExperienceWorker`** — manages context injection during rollout and cleanup during training.


### 1. Dynamic Experience Allocation

**Purpose**: 
Dynamically decide **how much and when** to use experience during both **training** and **rollout** stages.

**How it works**: 
This module performs two levels of adaptive allocation:

- **Task-Level Allocation**  
    - Determines whether each training task should **keep** or **discard** experience.
    - Controlled by `train_sample_mode`:  
        - `"allkeep"` → all tasks retain experience  
        - `"alldiscard"` → all tasks discard experience  
        - `"hybrid"` → keep ratio controlled by `train_sample_keepratio`
    - Key Codes:

```python
# Class: ExperienceManager
# Function: allocate_train_mode()
mode_to_ratio = {
    "allkeep": 1.0,
    "alldiscard": 0.0,
    "hybrid": self.train_sample_keepratio
}
keep_ratio = mode_to_ratio.get(
    self.train_sample_mode, self.train_sample_keepratio
)
keep_count = int(len(tasks) * keep_ratio)
exp_modes = ['keep'] * keep_count + ['discard'] * (len(tasks) - keep_count)
```

- **Rollout-Level Allocation**
    - Determines the proportion of rollouts within one task that will **include experience**.
    - Controlled by `val_rollout_mode` and `train_rollout_mode`:
        - `"woexp"` → no rollout uses experience (pure exploration) 
        - `"all"` → all rollouts use experience (fully guided) 
        - `"mixed"` → *partially guided*, rollout experience usage ratio determined by `rollout_ratio`
    - The parameter `rollout_ratio` only takes effect when `val_rollout_mode/train_rollout_mode="mixed"`. For example, `rollout_expratio=0.3` means *30%* of rollouts will include experience, while the remaining *70%* proceed without it.

```python
# Class: ExperienceManager
# Function: allocate_add_exp()
add_exp_choices = {
    "woexp": [False] * rollout_n,
    "mixed": sorted(
        [i < round(rollout_n * self.rollout_expratio) for i in range(rollout_n)],
        key=lambda _: random.random()
    ),
    "all": [True] * rollout_n
}[exp_mode]
```

**✅Effect**:

- `train_sample_mode` controls how experience is used in training samples.

-  `val_rollout_mode/train_rollout_mode` defines the exploration regime (`woexp` / `mixed` / `all`).

- `rollout_ratio` refines `mixed` mode by determining how many rollouts reuse experience.

Together, they enable dynamic balancing between exploration and exploitation.

### 2. Asynchronous Experience Summarization

**Purpose**: 
Convert raw trajectories into summarized experiences **asynchronously**, ensuring continuous learning without blocking.

**How it works**: 

- Periodically triggered by training steps (`updated_freq`).

- Executes summarization jobs via a background thread (`ThreadPoolExecutor`).

- Stores summarized results in the shared experience pool.

```python
# Class: ExperienceManager
# FUnction: submit_summary_task()
summary_task = self.thread_pool.submit(
    self.em_client.call_summarizer,
    trajectories=trajectories,
    workspace_id=reme_config.workspace_id
    )
```


**✅Effect**:
The experience pool grows in parallel with training, keeping the agent’s knowledge base continuously updated.

### 3. Context-Aware Rollout Management

**Purpose**: 
Make rollouts context-aware by injecting relevant past experiences into prompts.

**How it works**: 

- Retrieves top-K related experiences via `EMClient`.

- Formats and prepends them to the rollout message.

- Enhances rollout context without modifying the underlying task.

```python
# Class: ExperienceWorker
# Function: manage_rollout_context()
history_experience = self.em_client.call_context_generator(
    trajectory=trajectory,
    retrieve_top_k=reme_config.retrieve_top_k,
    workspace_id=reme_config.workspace_id
    )
formatted_experience = self.experience_template.format(history_experience)
new_content = formatted_experience + trajectory.steps[-1]["content"]
trajectory.steps[-1]["content"] = new_content
```


**✅Effect**:
Each rollout benefits from relevant prior knowledge, reducing redundant exploration.


### 4. Training Context Management

**Purpose**: 
Ensure training messages remain clean by removing injected experience when not needed.

**How it works**: 

- Detects experience templates in training messages using regex.

- Removes them when `train_mode="discard"` while retaining extracted text for analysis.


```python
# Class: ExperienceWorker
# Function: manage_training_context()
pattern = re.escape(self.experience_template).replace(r'\{\}', '(.*?)')
cleaned_message = re.sub(pattern, '', message, flags=re.DOTALL)
```


**✅Effect**:
Guarantees that training data integrity aligns with the current experience policy.




### 5. Training Loss Processing

**Purpose**:  
Ensure stable policy updates when mixing **on-policy rollouts** and **off-policy experience replays**, allowing the agent to leverage past trajectories without destabilizing learning.

**How it Works**:

- This module computes a *heterogeneous PPO loss* that combines:
    - **On-policy loss**: derived from fresh rollouts.  
    - **Off-policy loss**: derived from experience-augmented samples.  
- An **experience mask (`exp_mask`)** distinguishes the two. Each loss is clipped and optionally adjusted for negative advantages. Finally, the losses are combined and aggregated according to `loss_agg_mode`.

```python
# Function: het_compute_token_on_off_policy_loss()

# 1️⃣ Compute policy ratio and approximate KL divergence
negative_approx_kl = log_prob - old_log_prob  # difference between new and old log-probabilities
ratio = torch.exp(negative_approx_kl)       # policy ratio r_t = exp(log_pi_new - log_pi_old)
ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)  # approximate KL divergence

# 2️⃣ Compute on-policy losses (exp_mask = 0)
on_pg_losses, _, _ = compute_pg_losses(cliprange_low, cliprange_high)
# Mask out experience tokens to ensure only fresh rollouts contribute
on_pg_loss = verl_F.masked_mean(on_pg_losses, (1.0 - exp_mask) * response_mask)

# 3️⃣ Compute off-policy losses (exp_mask = 1)
off_pg_losses, _, _ = compute_pg_losses(off_cliprange_low, off_cliprange_high)
# Mask to include only experience tokens
off_pg_loss = verl_F.masked_mean(off_pg_losses, exp_mask * response_mask)
# Ensure numerical stability
off_pg_loss = torch.tensor(0.0) if off_pg_loss.isnan().item() else off_pg_loss

# 4️⃣ Combine both losses using the experience mask
pg_losses = off_pg_losses * exp_mask + on_pg_losses * (1.0 - exp_mask)
# Aggregate token-level losses according to selected mode (e.g., "token-mean")
pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
```

**✅Effect**:

- `exp_mask` separates on-policy and off-policy contributions cleanly.

- `off_cliprange_high` enforce trust regions for stable updates.


## ⚙️ Key Parameters & Configuration

### Rollout Modes

**`train_rollout_mode`** (*str*)  
: Controls how experiences are used during **training rollouts**.  
Options:  
- `"woexp"` → rollouts without any experience guidance (pure exploration)  
- `"mixed"` → partially inject experiences based on `rollout_ratio`  
- `"all"` → all rollouts include retrieved experiences  
Default: `"woexp"`.

**`val_rollout_mode`** (*str*)  
: Controls how experiences are used during **validation/test rollouts**.  
Same options as `train_rollout_mode`. Typically set to `"woexp"` for unbiased evaluation.  
Default: `"woexp"`.

**`rollout_ratio`** (*float*, range: [0.0, 1.0])  
: When rollout mode is `"mixed"`, this ratio determines the proportion of rollouts that include experience.  
Example: `0.3` means 30% experience-guided, 70% exploratory.  
Default: `0.0`.

---

### **Training Sample Processing**

**`train_sample_mode`** (*str*)  
: Defines whether to keep or discard experience context in **training samples** after rollout.  
Options:  
- `"allkeep"` → all samples retain experience information  
- `"alldiscard"` → strip experience from all samples (model learns from raw reasoning)  
- `"hybrid"` → selective retention based on `train_sample_keepratio`  
Default: `"alldiscard"`.

**`train_sample_keepratio`** (*float*, range: [0.0, 1.0])  
: When `train_sample_mode="hybrid"`, controls the **task-level** proportion of samples that retain experience.  
Default: `1.0`.

---

### **Experience Retrieval & Injection**

**`experience_template`** (*str*)  
: Template string for wrapping retrieved experiences before injecting into prompts.  
The `{}` placeholder is replaced with formatted experience content.  
Example: `"\n\nSome Related Experience to help you to complete the task:<EXP>{}</EXP>\n\n"`

**`retrieve_top_k`** (*int*)  
: Number of most relevant experiences to retrieve per query when `enable_context_generator=True`.  
Default: `3`.

---

### **ReMe Service Configuration**

**`base_url`** (*str*)  
: HTTP endpoint of the **ReMe service** (Reflective Memory Engine).  
Handles experience summarization, storage, and retrieval.  
Example: `"http://127.0.0.1:8001"`.

**`workspace_id`** (*str*)  
: Unique identifier for the experience workspace. Different workspaces maintain isolated experience pools.  
Default: `"default"`.

**`enable_summarizer`** (*bool*)  
: Whether to activate **automatic experience summarization** from rollout trajectories.  
When `True`, raw reasoning traces are distilled into reusable experience snippets.  
Default: `False`.

**`enable_context_generator`** (*bool*)  
: Whether to enable **experience retrieval and context injection** during rollouts.  
Must be `True` for experience-guided rollouts to function.  
Default: `False`.

**`updated_freq`** (*int*)  
: Frequency (in training steps) to refresh the experience pool.  
Set to `0` to disable periodic updates.  
Default: `0`.

---

### **Initialization & Utilities**

**`init_exp_before_training`** (*bool*)  
: Whether to **pre-populate the experience pool** before training begins.  
Useful for warm-start scenarios with prior knowledge.  
Default: `False`.

**`init_exp_only`** (*bool*)  
: If `True`, only initializes the experience pool without starting training.  
Useful for pre-computing embeddings or validating summarization quality.  
Default: `False`.

**`summary_batch_size`** (*int*)  
: Batch size for processing experience summarization requests.  
Larger values improve throughput but require more memory.  
Default: `8`.


## 🧭 Quick Start & Recommended Configuration

### Step 1: Set Up ReMe Service

1. Ensure you have completed the [ReMe installation](https://github.com/agentscope-ai/ReMe?tab=readme-ov-file#%EF%B8%8F-installation);

2. Ensure you have modified the [environment configuration](https://github.com/agentscope-ai/ReMe?tab=readme-ov-file#environment-configuration);

3. Start the ReMe service with the following command:

```bash
cd path_to_reme/ReMe

reme \
  config=default \
  backend=http \
  thread_pool_max_workers=256 \
  http.host="127.0.0.1" \
  http.port=8001 \
  http.limit_concurrency=256 \
  llm.default.model_name=qwen-max-2025-01-25 \
  embedding_model.default.model_name=text-embedding-v4 \
  vector_store.default.backend=local \
  op.rerank_memory_op.params.enable_llm_rerank=false \
  flow.summary_task_memory.flow_content="trajectory_preprocess_op >> (success_extraction_op|failure_extraction_op|comparative_extraction_op) >> memory_validation_op >> memory_deduplication_op >> update_vector_store_op"
```


### **Step 2: Recommended Configuration**
Add the following configuration to your experiment settings:
```yaml
exp_manager:
  val_rollout_mode: "woexp"    # rollout mode on dev/test set: ["mixed", "all", "woexp"]
  train_rollout_mode: "mixed"  # rollout mode on train set: ["mixed", "all", "woexp"]
  rollout_ratio: 0.5           # ratio of adding experience within group during rollout
  train_sample_mode: "alldiscard"    # sample mode after train rollout: ["allkeep", "alldiscard", "hybrid"]
  train_sample_keepratio: 0.0     # task-level ratio for keeping experience
  experience_template: "\n\nSome Related Experience to help you to complete the task:<EXP>{}</EXP>\n\n"  # template for inserting experience
  init_exp_before_training: True  # whether to init experience pool before training
  init_exp_only: False  # whether to only init experience pool
  summary_batch_size: 8  # batch size for experience summarization
  reme:
    base_url: "http://127.0.0.1:8001"  # base URL for ReMe service
    workspace_id: "default"  # workspace ID for ReMe
    enable_summarizer: True  # whether to enable experience summarizer
    enable_context_generator: True  # whether to enable context generator
    retrieve_top_k: 3  # number of top experiences to retrieve
    updated_freq: 0   # experience pool update frequency (in steps, 0=disabled)
```

### **Step 3: Understanding Different Usage Modes**

The experience pool system supports multiple usage modes to fit different experimental needs. Configure the parameters according to the table below:

#### Mode Descriptions

| Usage Mode | Description |
|---|---|
| **No Pool** | Training without any experience pool (baseline mode) |
| **Init Only** | Initialize experience pool from trajectories only, without using it during training |
| **Init + Train** | Initialize experience pool before training and use it throughout the training process |
| **Init + Train + Update** | Initialize experience pool, use during training, and continuously update it with new experiences |
| **Exist + Train** | Use an existing experience pool during training without initialization or updates |
| **Exist + Train + Update** | Use an existing experience pool during training and continuously update it with new experiences |



#### Configuration Matrix

|  | No<br>Pool | Init<br>Only | Init +<br>Train | Init + Train<br>+ Update | Exist +<br>Train | Exist + Train<br>+ Update |
|---|---|---|---|---|---|---|
| `updated_freq` | 0 | 0 | 0 | k != 0 | 0 | k != 0 |
| `init_exp_only` | False | True | False | False | False | False |
| `init_exp_before_training` | False | True | True | True | False | False |
| `enable_summarizer` | False | True | False | True | False | True |
| `enable_context_generator` | False | False | True | True | True | True |
| `workspace_id` | - | New ID | New ID | New ID | Exist ID | Exist ID |

#### Configuration Notes

- **`updated_freq`**: Set to a non-zero value (e.g., `100`) to enable periodic experience pool updates during training. `0` disables updates.

- **`workspace_id`**: 
  - Use a **new workspace ID** to create a fresh experience pool
  - Specify an **existing workspace ID** to reuse a previously created pool
  - When using `vector_store.default.backend=local`, the experience pool is saved at: `ReMe/local_vector_store/{workspace_id}.jsonl`

- **Recommended starting point**: For most use cases, we recommend starting with **`Init + Train`** mode.