<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10-blue" alt="Python 3.10"></a>
  <a href="./LICENSE.txt"><img src="https://img.shields.io/badge/license-Apache--2.0-black" alt="License"></a>
  <a href="https://modelscope.github.io/AgentEvolver/"><img src="https://img.shields.io/badge/docs-online-blue?logo=markdown" alt="Documentation"></a>
  <a href="https://github.com/modelscope/AgentEvolver"><img src="https://img.shields.io/badge/repository-AgentEvolver-181717?logo=github" alt="GitHub repository"></a>
  <a href="https://arxiv.org/abs/2602.06554"><img src="https://img.shields.io/badge/arXiv-2602.06554-b31b1b.svg" alt="arXiv"></a>
  <a href="https://deepwiki.com/modelscope/AgentEvolver"><img src="https://deepwiki.com/badge.svg" alt="deepwiki"></a>
  <a href="https://github.com/modelscope/AgentEvolver"><img src="https://img.shields.io/github/stars/modelscope/AgentEvolver?style=social" alt="GitHub Stars"></a>
</p>
<p align="center"><sub><strong>Branch</strong>: <code>seeupo</code> · <strong>Upstream</strong>: <a href="https://github.com/modelscope/AgentEvolver">modelscope/AgentEvolver</a> — Docs / repository / stars above follow that upstream (this README is the SeeUPO implementation track).</sub></p>

<p align="center">
  <a href="https://arxiv.org/pdf/2602.06554" title="PDF">
    <img src="https://img.shields.io/badge/PDF-Download-E74C3C?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="Download PDF"/>
  </a>
  <a href="LICENSE.txt" title="License">
    <img src="https://img.shields.io/badge/License-Apache%202.0-3DDC84?style=for-the-badge&logo=apache&logoColor=white" alt="Apache 2.0"/>
  </a>
</p>
<p align="center">
  <a href="https://www.python.org/" title="Python">
    <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10"/>
  </a>
  <a href="https://www.ray.io/" title="Ray">
    <img src="https://img.shields.io/badge/Distributed-Ray-028CF0?style=for-the-badge&logo=ray&logoColor=white" alt="Ray"/>
  </a>
  <a href="https://github.com/volcengine/verl" title="verl">
    <img src="https://img.shields.io/badge/Built%20on-verl-F97316?style=for-the-badge" alt="Built on verl"/>
  </a>
  <a href="https://doi.org/10.48550/arXiv.2602.06554" title="DOI">
    <img src="https://img.shields.io/badge/DOI-10.48550%2FarXiv.2602.06554-0066CC?style=for-the-badge" alt="DOI"/>
  </a>
</p>

<h1 align="center">
  <img src="https://github.com/QwenLM.png?size=96" width="44" height="44" alt="Tongyi Qwen"/>
  &nbsp;SeeUPO
</h1>

<p align="center">
  <a href="https://www.alibabagroup.com/en-US/" title="Alibaba Group">
    <img src="https://img.shields.io/badge/Alibaba-Group-FF6A00?style=for-the-badge&logo=alibabacloud&logoColor=white" alt="Alibaba Group"/>
  </a>
  &nbsp;
  <a href="https://tongyi.aliyun.com/" title="Tongyi">
    <img src="https://img.shields.io/badge/通义-Tongyi-7C3AED?style=for-the-badge" alt="通义 Tongyi"/>
  </a>
</p>

<p align="center">
  <strong>✨ Official implementation</strong> &nbsp;·&nbsp;
  <strong>🤖 Multi-turn agentic RL</strong> &nbsp;·&nbsp;
  <strong>⚡ Critic-free sequence-level updates</strong> &nbsp;·&nbsp;
  <strong>✅ Convergence guarantees</strong>
</p>

---

This repository implements the multi-turn agentic reinforcement learning training pipeline from **SeeUPO: Sequence-Level Agentic-RL with Convergence Guarantees**, built on the **BeyondAgent** framework and a **project-vendored [verl](https://github.com/volcengine/verl) tree under `external/verl/`** (this codebase **does not** use the `verl` package from PyPI). Training **requires** that copy: install it with **`pip install -e external/verl`** (see **Quick start (SeeUPO)** → **Environment setup**). A standalone **env_service** handles environment interaction.

## Contents

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

<p align="center">
  <a href="https://arxiv.org/abs/2602.06554" title="SeeUPO on arXiv">
    <img src="https://img.shields.io/badge/arXiv-2602.06554-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv 2602.06554"/>
  </a>
</p>

**Authors:** Tianyi Hu, Qingxu Fu, Yanxi Chen, Zhaoyang Liu, Bolin Ding  

**Submitted:** 2026-02-06 (arXiv:2602.06554)

### 🔬 What the paper proposes (and how it maps to this code)

> **TL;DR** — The paper studies **advantage estimation** (GAE vs. GRAE) × **policy updates** (REINFORCE vs. proximal / HAML-style). In **multi-turn** settings, common **critic-free** backbones lack **joint** critic-free + convergence guarantees; **SeeUPO** uses **reverse-order, turn-wise sequential updates** (HAML) so **backward induction** can target **global optimality**, still **without a critic**.

#### Takeaways (from §3 of the paper)

- **REINFORCE + GRAE** can converge to a **global optimum** under **undiscounted** (γ = 1) conditions; **PPO-style (PPU) + GRAE** generally **does not** keep the usual **monotonic improvement** story because of **structural bias** in the clipped objective.
- **Multi-turn** exposes a **trade-off**: mainstream recipes rarely achieve **both** **critic-free training** **and** strong **convergence-style guarantees**.
- **SeeUPO** treats a multi-turn trajectory as **sequential single-turn bandits / virtual agents**, updates **turn-by-turn in reverse order (T → T−1 → … → 1)**, and in practice instantiates **GRAE + PPO-style mirror updates** (paper: **SeeUPPO-GRAE**).

#### Table 1 (informal; from the paper): backbones × convergence sketch

**ST** = single-turn, **MT** = multi-turn. The sketch condenses claims from §3; **each entry is backed by formal analysis** (definitions, assumptions, lemmas/theorems, and proofs). For **full statements and derivations**, see **[arXiv:2602.06554](https://arxiv.org/abs/2602.06554) and the paper appendices** (e.g. GAE/GRAE bias, PPO-style objectives, multi-turn bandit modeling, HAML/monotonic improvement, backward induction / global optimality for SeeUPO).

<details>
<summary><strong>Table 1 — full grid</strong> (advantage / update / level / examples / ST–MT / repo mapping)</summary>

**Columns:** **Advantage** and **Update** name the estimator and policy-update family; **Level** is token vs. sequence; **Example** cites a representative method; **ST** / **MT** indicate whether the paper’s convergence sketch covers single-turn vs. multi-turn; **In this repo** points to config knobs when applicable.

| Advantage | Update | Level | Example | ST | MT | In this repo |
|:----------|:-------|:------|:--------|:--:|:--:|:----------|
| GAE | PPU (PPO-style) | Token | PPO | ✓ | ✓ | `adv_estimator: gae` (critic on) |
| GRAE | PPU | Token | GRPO, REINFORCE++ | ✗ | ✗ | `adv_estimator: grpo` (token-level baseline) |
| GRAE | REINFORCE | Sequence | RLOO | ✓ | ✗ | — |
| GRAE | PPU | Sequence | GSPO | ✓ | ✗ | `loss_mode: gspo` (sequence baseline) |
| GRAE | HAML / sequential | Sequence | **SeeUPO** | — | ✓ | `sequential_update`, `update_order: reverse`, `adv_updator: seeupo` |

</details>

**Configs in this repo:** `launcher/qwen3_appworld/`, `launcher/qwen3_bfcl/`, `launcher/qwen25_bfcl/` (YAML + shell helpers).

<a id="toc-repository-layout"></a>
## 🗂️ Repository layout

<details>
<summary><strong>Path → description</strong> (click to expand)</summary>

| Path | Description |
|:-----|:------------|
| `beyondagent/` | Main training loop and Ray trainer; `module/trainer/ba_ray_trainer.py` implements **SeeUPO-style sequential updates, ratio computation, and `adv_updator: seeupo`**. |
| `external/verl/` | **Required** SeeUPO/BeyondAgent fork of verl shipped in-repo — **not** interchangeable with **`pip install verl`** from PyPI. After cloning this repo, install with **`pip install -e external/verl`** (optionally `--no-deps` per **Environment setup**). Use the revision bundled here; other forks/commits may break training. |
| `env_service/` | Environment service for **AppWorld**, **BFCL**, **OpenWorld**, etc.; launch scripts live in `env_service/launch_script/`. |
| `launcher/` | Hydra/YAML experiment entry points; e.g. SeeUPO on BFCL: `launcher/qwen3_bfcl/qwen3-seeupo-bfcl.yaml`. |
| `config/` | Shared Hydra fragments for defaults and dataflow. |
| `seeupo_env.yaml` | Exported Conda environment for dependency pinning. |
| `requirements_NewVerl.txt` | Ultra-short install reminder; the English walkthrough lives under **Quick start (SeeUPO)** / **Environment setup** below. |
| `sync_env_with_yaml.py` | Compare / align an activated Conda env with `seeupo_env.yaml` (strict version sync). |

</details>

HTTP API details for environments are documented in `env_service/interface.md` (ports depend on your setup; training YAMLs typically set `env_service.env_url`).

<a id="toc-quick-start-seeupo"></a>
## 🚀 Quick start (SeeUPO)

Use this as the entry point: it points to the **same** end-to-end steps as **Environment setup** immediately below — benchmark sandboxes **(A)** plus the training Conda stack **(B)** with **`pip install -e external/verl`**, FlashAttention, vLLM, and optional **`sync_env_with_yaml.py`**. Follow **Environment setup** for commands and version pins; nothing here replaces that section.

<a id="toc-environment-setup"></a>
## 🧰 Environment setup

Set things up in two layers: **(A)** per-benchmark sandboxes under `env_service/environments/`, and **(B)** the **agentic RL training stack** (Python, **this repo’s `external/verl`**, FlashAttention, vLLM) used by `launcher.py`.

**verl (mandatory):** You must use the **verl sources included in this repository** (`external/verl/`). Do **not** install verl from PyPI or substitute another Git clone unless it matches the project-pinned tree. The supported install is **`pip install -e external/verl`** from the repo root (see step 3 below; `--no-deps` is recommended so Conda/`seeupo_env.yaml` control dependencies).

### A) `env_service` benchmark sandboxes

Each benchmark ships a small **`setup.sh`** that installs or prepares its local dependencies (datasets, Python env hints, etc.). From the repo root, run the script for the benchmark you need:

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

Read any messages the script prints (paths, extra Conda envs, data downloads). **BFCL** in particular may require running preprocessing steps mentioned inside `env_service/launch_script/bfcl.sh` / the BFCL README so that `BFCL_DATA_PATH` and related files exist. After setup, start the HTTP env service with `env_service/launch_script/appworld.sh`, `bfcl.sh`, etc., or use the launcher shell scripts in `launcher/`, which start the service for you.

### B) Agentic RL infrastructure (training Conda env)

The **canonical version pin** is **`seeupo_env.yaml`** (a full Conda export). The steps below are the recommended **high-level recipe**; if anything conflicts, **prefer the exact versions in `seeupo_env.yaml`**.

Same intent as `requirements_NewVerl.txt`; **exact pins** are in **`seeupo_env.yaml`**.

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

The snippets below track the current **`launcher/qwen3_bfcl/qwen3-seeupo-bfcl.yaml`**. Other benchmarks mirror the same **`algorithm`** block; adjust **`env_service`**, **`trainer.nnodes`**, paths, and model checkpoints for your run.

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

Set **`default_local_dir`**, **`experiment_name`**, **`n_gpus_per_node`**, **`nnodes`**, **`total_epochs`**, loggers (`swanlab`, etc.). BFCL reference run: **50 epochs**, **8×1 GPUs** in the checked-in YAML.

### `actor_rollout_ref` — optimization, rollout, model

- **`actor`:** LR **1e-6**, **KL** penalty (`kl_loss_coef: 0.002`, `low_var_kl`), **FSDP offload** flags, **dynamic batch** tokens (`ppo_max_token_len_per_gpu`, etc.).
- **`rollout`:** **`vllm`** + **`mode: async`**, **`n: 8`** rollouts per prompt, **`multi_turn.max_steps: 10`**, temperature **0.9**, **`context_template: linear_think`** (SeeUPO family uses linear thinking template), lengths aligned to data (`prompt_length` / `response_length` / `max_model_len`).
- **`model`:** set **`path`** to your **Qwen3** checkpoint; **`use_qwen3: True`**, gradient checkpointing / padding as in the YAML.
- **`critic`:** for critic-free runs, keep the **`critic.model`** block **commented** as in the file.

For AppWorld, copy the same **`algorithm`** block and point **`env_service.env_type`** / URLs to the AppWorld service; see `launcher/qwen3_appworld/qwen3-seeupo-appworld.yaml`.

<a id="toc-run-training"></a>
## 🚀 Run training

**Prerequisites:** complete **Quick start (SeeUPO)** / **Environment setup** (benchmark sandboxes + training Conda env, optionally **`sync_env_with_yaml.py`**) in the sections above. Activate your **training** environment (`conda activate seeupo` or your name), `cd` to the repo root, and ensure CUDA/driver match your vLLM build.

### Single-node (GRPO / GSPO / SeeUPO)

The scripts under **`launcher/qwen3_bfcl/`** and **`launcher/qwen3_appworld/`** are the fastest path: start the env service with **`nohup`**, wait, then call **`launcher.py`** with the matching YAML.

<details>
<summary><strong>Launcher scripts</strong> (environment × GRPO / GSPO / SeeUPO)</summary>

| Environment | GRPO | GSPO | SeeUPO |
|:------------|:-----|:-----|:-------|
| BFCL | `bash launcher/qwen3_bfcl/qwen3-grpo-bfcl.sh` | `bash launcher/qwen3_bfcl/qwen3-gspo-bfcl.sh` | `bash launcher/qwen3_bfcl/qwen3-seeupo-bfcl.sh` |
| AppWorld | `bash launcher/qwen3_appworld/qwen3-grpo-appworld.sh` | `bash launcher/qwen3_appworld/qwen3-gspo-appworld.sh` | `bash launcher/qwen3_appworld/qwen3-seeupo-appworld.sh` |

</details>

**Required env vars:** **`CONDA_SH`**, **`SWANLAB_API_KEY`**. **Optional:** **`BFCL_CONDA_ENV`** / **`APPWORLD_CONDA_ENV`** (defaults `bfcl` / `appworld`), **`TRAIN_CONDA_ENV`** (default `seeupo`), **`BFCL_ENV_DIR`**, **`BFCL_STARTUP_SLEEP`** / **`APPWORLD_STARTUP_SLEEP`**, **`APPWORLD_ROOT`**. If you use the context-template alien LLM path, set **`DASHSCOPE_API_KEY`** (or **`DASHSCOPE_API_KEYS`** / **`DASHSCOPE_API_KEYS_REGULAR`** + **`DASHSCOPE_API_KEYS_BACKUP`** as comma-separated lists). Details are in the header comments of each script.

**Logs:** `bfcl_service.log` / `appworld_service.log` at the **repository root**.

### Manual flow

If you already run the env service by hand, skip the `nohup` block and run from the repo root:

<details>
<summary><strong>Example — manual <code>launcher.py</code> (BFCL SeeUPO)</strong></summary>

```bash
python launcher.py --conf launcher/qwen3_bfcl/qwen3-seeupo-bfcl.yaml
```

</details>

For AppWorld you may also use **`python launcher.py --conf <yaml> --with-appworld`** so the launcher starts AppWorld instead of a pre-started service.

### Multi-node (PPO baseline)

Use **`launcher_multinode.py`** with **`launcher/qwen3_bfcl/qwen3-ppo-bfcl.sh`** or **`launcher/qwen3_appworld/qwen3-ppo-appworld.sh`**. Your scheduler must set **`RANK`**, **`WORLD_SIZE`**, **`MASTER_ADDR`**, **`MASTER_PORT`**, plus **`CONDA_SH`** and **`SWANLAB_API_KEY`**. Optional: **`TRAIN_CONDA_ENV`**, **`BFCL_CONDA_ENV`** / **`APPWORLD_CONDA_ENV`**, **`NUM_GPUS_PER_NODE`**, **`NUM_CPUS_PER_NODE`**, **`OBJECT_STORE_MEMORY`**, **`NCCL_*` / `GLOO_*`**. Service logs: **`logs/bfcl/`** / **`logs/appworld/`** under the repo root.

`launcher.py` backs up `config/`, `beyondagent/`, and the chosen YAML under the experiment directory for reproducibility.

---

<a id="toc-license"></a>
## 📜 License

`LICENSE.txt` in this repository is **Apache License 2.0** (see the file for the full text).

<a id="toc-citation"></a>
## 📚 Citation (BibTeX)

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
