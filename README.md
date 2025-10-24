<p align="center">
 <img src="docs/img/logo.png" alt="AgentEvolver Logo" width="70%">
</p>
<h2 align="center">AgentEvolver: Towards Efficient Self-Evolving Agent System</h2>

<!-- --- -->

<p align="center">
  <a href="https://arxiv.org/abs/0000"><img src="https://img.shields.io/badge/cs.MA-0000-B31C1C?logo=arxiv&logoColor=B31C1C" alt="arxiv"/></a>
  <a href="https://pypi.org/project/reme-ai/"><img src="https://img.shields.io/badge/python-3.12+-blue" alt="Python Version"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-black" alt="License"></a>
  <a href="https://github.com/modelscope/AgentEvolver"><img src="https://img.shields.io/github/stars/modelscope/AgentEvolver?style=social" alt="GitHub Stars"></a>
</p>



<!-- <p align="center">
  <strong>AgentEvolver: An Efficient Self-Evolving Agent System</strong><br>
</p> -->

**AgentEvolver** is an end-to-end, self-evolving training framework that unifies self-questioning, self-navigating, and self-attributing into a cohesive system. It empowers agents to autonomously
improve their capabilities, aiming for efficient, cost-effective, and continuous capability evolution.


## 📰 News

- **[2025-10]** 📄 The AgentEvolver Technical Report is now available, detailing the framework’s architecture, methodology, and key findings.
- **[2025-10]** 🧩 AgentEvolver v1 has been released now!


## ✨ Why AgentEvolver

<p align="center">
 <img src="docs/img/flowchart.png" alt="AgentEvolver Logo" width="80%">
</p>

🧠 Three **Self-Evolving Mechanisms** from Environment to Policy:

- **Automatic Task Generation (Self-Questioning)** – Explore the environment and autonomously create diverse tasks, eliminating costly manual dataset construction.
- **Experience-guided Exploration (Self-Navigating)** – Summarize and reuse cross-task experience, guiding higher-quality rollouts and improving exploration efficiency.
- **Attribution-based Credit Assignment (Self-Attributing)** – Process long trajectories to uncover the causal contribution of intermediate steps, enabling fine-grained and efficient policy optimization.




## 🔧 Architecture Design
AgentEvolver adopts a service-oriented dataflow architecture, seamlessly integrating environment sandboxes, LLMs, and experience management into modular services.

<p align="center">
 <img src="docs/img/system.png" alt="system framework" width="80%">
</p>


- **Environment Compatibility** – Standardized interfaces for seamless integration with a wide range of external environments and tool APIs.
- **Flexible Context Manager** – Built-in utilities for managing multi-turn contexts and complex interaction logic, supporting diverse deployment scenarios.
- **Modular & Extensible Architecture** – Decoupled components allow easy customization, secondary development, and future algorithm upgrades.




## 🚀 Quick Start
### Step 1. Basic Dependency Installation

First, clone all submodule.
```bash
git submodule update --init external/verl
```

Then, set up the training environment, choose between `uv` and `conda`.

<details>
<summary>🛠️ Set up environment with uv (Click to read detail)</summary>

```bash
# 🧰 setup uv (you can also choose conda if you prefer, but conda is too slow)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python=3.11 # If this step is slow, add ENV variable: UV_PYTHON_INSTALL_MIRROR="https://gh-proxy.com/https://github.com/astral-sh/python-build-standalone/releases/download"
source .venv/bin/activate
# 🌱 clone our verl branch
git submodule update --init external/verl
# 🆙 make sure our pip is ready
uv pip install --upgrade pip setuptools packaging -i https://mirrors.aliyun.com/pypi/simple/
uv pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --no-deps --prerelease=allow
uv pip install -e external/verl -i https://mirrors.aliyun.com/pypi/simple/
# ✨ finally, install flash attention (must be installed at last, need to connect to github)
uv pip install --verbose flash-attn==2.7.4.post1 ring-flash-attn -i https://mirrors.aliyun.com/pypi/simple/ --no-deps --no-build-isolation
```

</details>
<details>
<summary>🛠️ Set up environment with conda (Click to read detail)</summary>

```bash
conda create -n appworld python=3.11 -y
conda activate appworld
# 🆙 make sure our pip is ready
pip install --upgrade pip setuptools packaging -i https://mirrors.aliyun.com/pypi/simple/
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --no-deps
pip install -e external/verl -i https://mirrors.aliyun.com/pypi/simple/
pip install --verbose flash-attn==2.7.4.post1 ring-flash-attn -i https://mirrors.aliyun.com/pypi/simple/ --no-deps --no-build-isolation
```

</details>

### Step 2. Setup Env-Service (Appworld as example)
The script below sets up an environment for appworld. For other environment setup, refer to [docs/guidelines/env_service.md](docs/guidelines/env_service.md) 📄

```bash
cd env_service/environments/appworld && bash setup.sh
```

### Step 3. Begin Training! 🚀 🚀

Using AgentEvoler launcher to start environment, log dashboard and training process altogether. Refer to [docs/launcher.md](docs/launcher.md) for more launcher information.

```bash
python launcher.py --conf examples/self-question-attr.yaml --with-appworld --with-logview
```
