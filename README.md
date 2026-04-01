<p align="center">
  <a href="https://modelscope.github.io/AgentEvolver/"><img src="https://img.shields.io/badge/docs-online-blue?logo=markdown" alt="Documentation"></a>
  <a href="https://github.com/modelscope/AgentEvolver"><img src="https://img.shields.io/badge/repository-AgentEvolver-181717?logo=github" alt="GitHub repository"></a>
  <a href="https://arxiv.org/abs/2602.06554"><img src="https://img.shields.io/badge/arXiv-2602.06554-b31b1b.svg" alt="arXiv"></a>
  <a href="https://deepwiki.com/modelscope/AgentEvolver"><img src="https://deepwiki.com/badge.svg" alt="deepwiki"></a>
  <a href="https://github.com/modelscope/AgentEvolver"><img src="https://img.shields.io/github/stars/modelscope/AgentEvolver?style=social" alt="GitHub Stars"></a>
</p>

<h1 align="center">
  <img src="https://github.com/QwenLM.png?size=96" width="44" height="44" alt="Tongyi Qwen" style="vertical-align: middle;"/>
  &nbsp;SeeUPO
</h1>

<p align="center"><em>Sequence-Level Agentic-RL with Convergence Guarantees</em></p>

<p align="center">
  <strong>✨ Multi-turn Agentic RL</strong> &nbsp;·&nbsp;
  <strong>⚡ Critic-free Sequence-level Updates</strong> &nbsp;·&nbsp;
  <strong>✅ Convergence Guarantees</strong>
</p>

<p align="center">
  <a href="https://www.alibabagroup.com/en-US/" title="Alibaba Group">
    <img src="https://img.shields.io/badge/Alibaba-Group-FF6A00?style=for-the-badge&logo=alibabacloud&logoColor=white" alt="Alibaba Group"/>
  </a>
  &nbsp;
  <a href="https://tongyi.aliyun.com/" title="Tongyi">
    <img src="https://img.shields.io/badge/通义-Tongyi-7C3AED?style=for-the-badge" alt="通义 Tongyi"/>
  </a>
  &nbsp;
  <a href="https://arxiv.org/pdf/2602.06554" title="PDF">
    <img src="https://img.shields.io/badge/PDF-Download-E74C3C?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="Download PDF"/>
  </a>
</p>

<p align="center"><sub><strong>Implementation track</strong>: <code>seeupo</code> · <strong>Upstream</strong>: <a href="https://github.com/modelscope/AgentEvolver">modelscope/AgentEvolver</a></sub></p>

---

> **SeeUPO** is a multi-turn agentic reinforcement learning training pipeline built on **BeyondAgent** with a **modified and vendored verl** under `external/verl/`.
> This verl has been approved for open source. Training **requires** using this vendored copy — install it with `pip install -e external/verl --no-deps`. Environment interaction is served by the standalone **`env_service`**.

## Contents

Use the roadmap below to jump directly to the paper summary, repository structure, setup instructions, or training entry points.

- [Paper](#toc-paper)
- [Repository layout](#toc-repository-layout)
- [Quick start (SeeUPO)](#toc-quick-start-seeupo)
- [Environment setup](#toc-environment-setup)
- [SeeUPO-related training settings](#toc-seeupo-training-settings)
- [Run training](#toc-run-training)
- [License](#toc-license)
- [Citation (BibTeX)](#toc-citation)

<a id="toc-paper"></a>
## 📄 Paper

**Authors:** Tianyi Hu, Qingxu Fu, Yanxi Chen, Zhaoyang Liu, Bolin Ding  
**Submitted:** 2026-02-06 &nbsp;|&nbsp; [arXiv:2602.06554](https://arxiv.org/abs/2602.06554)

### 🔬 What the paper proposes (and how it maps to this code)

> **TL;DR.** The paper studies **advantage estimation** (GAE vs. GRAE) together with **policy updates** (REINFORCE vs. proximal / HAML-style). In **multi-turn** settings, standard **critic-free** recipes usually do not provide both critic-free training and convergence guarantees. **SeeUPO** addresses this with **reverse-order, turn-wise sequential updates** so that **backward induction** can target **global optimality** while remaining **critic-free**.

#### Takeaways (from §3 of the paper)

- **REINFORCE + GRAE** can converge to a **global optimum** under **undiscounted** (γ = 1) conditions; **PPO-style (PPU) + GRAE** generally **does not** keep the usual **monotonic improvement** story because of **structural bias** in the clipped objective.
- **Multi-turn** exposes a **trade-off**: mainstream recipes rarely achieve **both** **critic-free training** **and** strong **convergence-style guarantees**.
- **SeeUPO** treats a multi-turn trajectory as **sequential single-turn bandits / virtual agents**, updates **turn-by-turn in reverse order (T → T−1 → … → 1)**, and in practice instantiates **GRAE + PPO-style mirror updates** (paper: **SeeUPPO-GRAE**).

#### Table 1. Backbones and convergence sketch

**ST** = single-turn, **MT** = multi-turn. The table below condenses the claims in §3. **Each entry is backed by formal analysis** in the paper and appendices, including definitions, assumptions, lemmas, theorems, and proofs.

**Reading guide:** **Advantage** and **Update** name the estimator and policy-update family; **Level** is token vs. sequence; **Example** gives a representative method; **ST** / **MT** indicate whether the paper’s convergence sketch covers single-turn vs. multi-turn; **In this repo** points to relevant config knobs.

| Advantage | Update | Level | Example | ST | MT | In this repo |
|:----------|:-------|:------|:--------|:--:|:--:|:----------|
| GAE | PPU (PPO-style) | Token | PPO | ✓ | ✓ | `adv_estimator: gae` (critic on) |
| GRAE | PPU | Token | GRPO, REINFORCE++ | ✗ | ✗ | `adv_estimator: grpo` (token-level baseline) |
| GRAE | REINFORCE | Sequence | RLOO | ✓ | ✗ | — |
| GRAE | PPU | Sequence | GSPO | ✓ | ✗ | `loss_mode: gspo` (sequence baseline) |
| GRAE | HAML / sequential | Sequence | **SeeUPO** | — | ✓ | `sequential_update`, `update_order: reverse`, `adv_updator: seeupo` |

**Configs in this repo:** `launcher/qwen3_appworld/`, `launcher/qwen3_bfcl/`, `launcher/qwen25_bfcl/` (YAML + shell helpers).

<a id="toc-repository-layout"></a>
## 🗂️ Repository layout

The repository is organized around four pieces: the training core, the vendored `verl` dependency, the environment service, and benchmark-specific launch/config files.

| Path | Description |
|:-----|:------------|
| `beyondagent/` | Main training loop and Ray trainer; `module/trainer/ba_ray_trainer.py` implements **SeeUPO-style sequential updates, ratio computation, and `adv_updator: seeupo`**. |
| `external/verl/` | **Required** SeeUPO modified version of verl (approved for open source). Install with `pip install -e external/verl --no-deps`. You must use the version included here. |
| `env_service/` | Environment service for **AppWorld**, **BFCL**, **OpenWorld**, etc.; launch scripts live in `env_service/launch_script/`. |
| `launcher/` | Hydra/YAML experiment entry points; e.g. SeeUPO on BFCL: `launcher/qwen3_bfcl/qwen3-seeupo-bfcl.yaml`. |
| `config/` | Shared Hydra fragments for defaults and dataflow. |
| `seeupo_env.yaml` | Exported Conda environment for dependency pinning. |
| `requirements_NewVerl.txt` | Ultra-short install reminder; the English walkthrough lives under **Quick start (SeeUPO)** / **Environment setup** below. |
| `sync_env_with_yaml.py` | Compare / align an activated Conda env with `seeupo_env.yaml` (strict version sync). |

HTTP API details for environments are documented in `env_service/interface.md` (ports depend on your setup; training YAMLs typically set `env_service.env_url`).

<a id="toc-quick-start-seeupo"></a>
## 🚀 Quick start (SeeUPO)

Use this section as the shortest path into the project. The full setup still has **two layers**: **(A)** benchmark sandboxes under `env_service/environments/` and **(B)** the training Conda stack with **`pip install -e external/verl`**, FlashAttention, vLLM, and optional **`sync_env_with_yaml.py`**. Exact commands and version pins are given in **Environment setup** below.

<a id="toc-environment-setup"></a>
## 🧰 Environment setup

Set up the project in **two layers**:

- **(A) Benchmark sandboxes:** local benchmark dependencies under `env_service/environments/`
- **(B) Training infrastructure:** Python plus **this repo’s `external/verl`**, FlashAttention, and vLLM for `launcher.py`

> **Important.** You must use the **modified verl** vendored in this repository at `external/verl/`. This version has been reviewed and approved for open-sourcing. Do **not** install `verl` from PyPI or use another clone. Run `pip install -e external/verl --no-deps`.

### A) `env_service` benchmark sandboxes

Each benchmark provides a small **`setup.sh`** that prepares its local dependencies, datasets, and environment hints. From the repo root, run the script for the benchmark you need:

<details>
<summary><strong>Benchmark <code>setup.sh</code> commands</strong></summary>

```bash
# AppWorld
bash env_service/environments/appworld/setup.sh

# BFCL
bash env_service/environments/bfcl/setup.sh

# OpenWorld (optional)
bash env_service/environments/openworld/setup.sh
```

</details>

**After setup:** read the script output carefully for paths, extra Conda envs, and data downloads. **BFCL** may additionally require preprocessing steps referenced in `env_service/launch_script/bfcl.sh` or the BFCL README so that `BFCL_DATA_PATH` and related files exist. Once ready, start the HTTP environment service with `env_service/launch_script/appworld.sh`, `bfcl.sh`, and related scripts, or let the launcher scripts in `launcher/` start it for you.

### B) Agentic RL infrastructure (training Conda env)

The **canonical version pin** is **`seeupo_env.yaml`**. The recipe below is the recommended high-level path; if anything conflicts, **prefer the exact versions in `seeupo_env.yaml`**. This serves the same purpose as `requirements_NewVerl.txt`, but the YAML is the authoritative source of exact pins.

<details>
<summary><strong>Conda env recipe</strong> (create → <code>seeupo_env.yaml</code> → editable <code>external/verl</code> → flash-attn / vLLM → <code>sync_env_with_yaml.py</code>)</summary>

```bash
# 1) Create and activate a Conda env (Python 3.10)
conda create -n seeupo python=3.10
conda activate seeupo

# 2) Install most dependencies from seeupo_env.yaml (verl, flash-attn, vllm are added in step 3)
conda env update -f seeupo_env.yaml -n seeupo

# 3) Editable install of THIS REPO's verl (required — not pip install verl from PyPI), then FlashAttention and vLLM
pip install -e external/verl --no-deps
pip install flash-attn==2.7.0.post2 --no-deps --no-build-isolation
pip install vllm==0.8.5

# 4) Strict sync with seeupo_env.yaml (YAML-listed packages only); on resolver/ABI issues, align versions from the YAML instead of ad hoc bumps
python sync_env_with_yaml.py seeupo_env.yaml -n seeupo --compare-only
python sync_env_with_yaml.py seeupo_env.yaml -n seeupo --install
```

</details>

`sync_env_with_yaml.py` compares your environment to the YAML, then optionally installs mismatches. Other flags: **`--pip-only`**, **`--conda-only`**, **`--env-update`** (full `conda env update` from the YAML). If you omit the YAML path, the script falls back to **`seeupo_env.yaml`** next to the script when the default path is missing.

<a id="toc-seeupo-training-settings"></a>
## ⚙️ SeeUPO-related training settings

The snippets below summarize the checked-in **`launcher/qwen3_bfcl/qwen3-seeupo-bfcl.yaml`**. Other benchmarks reuse the same **`algorithm`** block; in practice you mainly adjust **`env_service`**, **`trainer.nnodes`**, dataset paths, and model checkpoints.

### Algorithm (`algorithm`)

<details>
<summary><strong>YAML — <code>algorithm</code></strong> (advantage, loss, sequential SeeUPO core)</summary>

```yaml
algorithm:
  # (1) GRAE-family: use GRPO-style advantage
  adv_estimator: grpo
  use_kl_in_reward: False

  # (2) Sequence-level policy loss (GSPO)
  loss_mode: gspo
  loss_agg_mode: "seq-mean-token-mean"

  # (3) Turn-wise sequential updates (SeeUPO schedule)
  sequential_update: True
  update_order: "reverse"   # sequential | reverse | random | custom

  # (4) SeeUPO advantage correction (mirror / IS terms); set None to disable
  adv_updator: seeupo
  adv_clip_ratio_high: 0.2
  adv_clip_ratio_low: 0.2

  # (5) Advantage normalization (batch-level; paper §4.2 / §5.3.2)
  norm_adv_by_std_in_grpo: False   # group std norm off (convergence-sensitive)
  special_norm: True               # enables batch-level normalization path in code
```

</details>

### Environment service (`env_service`)

<details>
<summary><strong>YAML — <code>env_service</code></strong> (where rollouts talk to the benchmark)</summary>

```yaml
env_service:
  env_url: http://localhost:8080
  env_type: bfcl
  env_feedin_preference: code
```

</details>

### Data (`data`)

<details>
<summary><strong>YAML — <code>data</code></strong> (batching and file paths)</summary>

```yaml
data:
  train_batch_size: 32
  max_prompt_length: 14000
  max_response_length: 4000
  return_raw_chat: True
  train_files: '//external/bfcl/data/BFCL_v4_multi_turn_base_train.parquet'
  val_files: '//external/bfcl/data/BFCL_v4_multi_turn_base_test.parquet'
```

</details>

### `trainer` — hardware & logging (abridged)

Use this block to set hardware scale, experiment naming, and logging. In particular, configure **`default_local_dir`**, **`experiment_name`**, **`n_gpus_per_node`**, **`nnodes`**, **`total_epochs`**, and your loggers (`swanlab`, etc.). The checked-in BFCL reference run uses **50 epochs** on **8×1 GPUs**.

### `actor_rollout_ref` — optimization, rollout, model

This block controls optimization, rollout behavior, and model wiring for training.

- **`actor`:** LR **1e-6**, **KL** penalty (`kl_loss_coef: 0.002`, `low_var_kl`), **FSDP offload** flags, **dynamic batch** tokens (`ppo_max_token_len_per_gpu`, etc.).
- **`rollout`:** **`vllm`** + **`mode: async`**, **`n: 8`** rollouts per prompt, **`multi_turn.max_steps: 10`**, temperature **0.9**, **`context_template: linear_think`** (SeeUPO family uses linear thinking template), lengths aligned to data (`prompt_length` / `response_length` / `max_model_len`).
- **`model`:** set **`path`** to your **Qwen3** checkpoint; **`use_qwen3: True`**, gradient checkpointing / padding as in the YAML.
- **`critic`:** for critic-free runs, keep the **`critic.model`** block **commented** as in the file.

For AppWorld, keep the same **`algorithm`** block and switch **`env_service.env_type`** and URLs to the AppWorld service; see `launcher/qwen3_appworld/qwen3-seeupo-appworld.yaml`.

<a id="toc-run-training"></a>
## 🚀 Run training

> **Prerequisites.** Complete **Quick start (SeeUPO)** / **Environment setup** first, including benchmark sandboxes, the training Conda environment, and optionally **`sync_env_with_yaml.py`**. Before launching, activate your training environment (`conda activate seeupo` or your own name), `cd` to the repo root, and verify that your CUDA/driver stack matches the installed vLLM build.

### Single-node (GRPO / GSPO / SeeUPO)

This is the fastest path for single-node experiments. The scripts under **`launcher/qwen3_bfcl/`** and **`launcher/qwen3_appworld/`** start the environment service with **`nohup`**, wait for readiness, and then invoke **`launcher.py`** with the matching YAML.

| Environment | GRPO | GSPO | SeeUPO |
|:------------|:-----|:-----|:-------|
| BFCL | `bash launcher/qwen3_bfcl/qwen3-grpo-bfcl.sh` | `bash launcher/qwen3_bfcl/qwen3-gspo-bfcl.sh` | `bash launcher/qwen3_bfcl/qwen3-seeupo-bfcl.sh` |
| AppWorld | `bash launcher/qwen3_appworld/qwen3-grpo-appworld.sh` | `bash launcher/qwen3_appworld/qwen3-gspo-appworld.sh` | `bash launcher/qwen3_appworld/qwen3-seeupo-appworld.sh` |

**Required env vars:** **`CONDA_SH`**, **`SWANLAB_API_KEY`**.  
**Optional env vars:** **`BFCL_CONDA_ENV`** / **`APPWORLD_CONDA_ENV`** (defaults `bfcl` / `appworld`), **`TRAIN_CONDA_ENV`** (default `seeupo`), **`BFCL_ENV_DIR`**, **`BFCL_STARTUP_SLEEP`** / **`APPWORLD_STARTUP_SLEEP`**, **`APPWORLD_ROOT`**.  
If you use the context-template alien LLM path, set **`DASHSCOPE_API_KEY`** or provide **`DASHSCOPE_API_KEYS`** / **`DASHSCOPE_API_KEYS_REGULAR`** together with **`DASHSCOPE_API_KEYS_BACKUP`** as comma-separated lists. Full details are documented in the header comments of each script.

**Logs:** `bfcl_service.log` / `appworld_service.log` at the **repository root**.

### Manual flow

Use this path if the environment service is already running. In that case, skip the launcher script’s `nohup` block and call `launcher.py` directly from the repo root:

<details>
<summary><strong>Example — manual <code>launcher.py</code> (BFCL SeeUPO)</strong></summary>

```bash
python launcher.py --conf launcher/qwen3_bfcl/qwen3-seeupo-bfcl.yaml
```

</details>

For AppWorld, you may also use **`python launcher.py --conf <yaml> --with-appworld`** so the launcher starts AppWorld instead of expecting a pre-started service.

### Multi-node (PPO baseline)

Use **`launcher_multinode.py`** together with **`launcher/qwen3_bfcl/qwen3-ppo-bfcl.sh`** or **`launcher/qwen3_appworld/qwen3-ppo-appworld.sh`** for distributed PPO baselines. Your scheduler must provide **`RANK`**, **`WORLD_SIZE`**, **`MASTER_ADDR`**, **`MASTER_PORT`**, **`CONDA_SH`**, and **`SWANLAB_API_KEY`**. Optional knobs include **`TRAIN_CONDA_ENV`**, **`BFCL_CONDA_ENV`** / **`APPWORLD_CONDA_ENV`**, **`NUM_GPUS_PER_NODE`**, **`NUM_CPUS_PER_NODE`**, **`OBJECT_STORE_MEMORY`**, and **`NCCL_*` / `GLOO_*`**. Service logs are written under **`logs/bfcl/`** or **`logs/appworld/`** at the repo root.

`launcher.py` backs up `config/`, `beyondagent/`, and the chosen YAML under the experiment directory for reproducibility.

---

<a id="toc-license"></a>
## 📜 License

This repository is released under **Apache License 2.0**. See `LICENSE.txt` for the full text.

<a id="toc-citation"></a>
## 📚 Citation (BibTeX)

Use the following BibTeX entry to cite the paper.

<details>
<summary><strong>BibTeX</strong> (click to expand)</summary>

```bibtex
@article{hu2026seeupo,
  title={SeeUPO: Sequence-Level Agentic-RL with Convergence Guarantees},
  author={Hu, Tianyi and Fu, Qingxu and Chen, Yanxi and Liu, Zhaoyang and Ding, Bolin},
  journal={arXiv preprint arXiv:2602.06554},
  year={2026},
  url={https://arxiv.org/abs/2602.06554}
}
```

</details>

You can also export BibTeX from the [arXiv abstract page](https://arxiv.org/abs/2602.06554).
