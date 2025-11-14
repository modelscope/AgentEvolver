<p align="center">
 <img src="docs/img/logo.png" alt="AgentEvolver Logo" width="70%">
</p>
<h2 align="center">AgentEvolver: Towards Efficient Self-Evolving Agent System</h2>

<!-- --- -->

<p align="center">
  <!-- <a href="https://arxiv.org/abs/0000"><img src="https://img.shields.io/badge/cs.MA-0000-B31C1C?logo=arxiv&logoColor=B31C1C" alt="arxiv"/></a> -->
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11+-blue" alt="Python Version"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-black" alt="License"></a>
  <a href="https://modelscope.github.io/AgentEvolver/"><img src="https://img.shields.io/badge/docs-online-blue?logo=markdown" alt="Documentation"></a>
  <a href="https://arxiv.org/abs/2511.10395"><img src="https://img.shields.io/badge/arXiv-2511.10395-b31b1b.svg" alt="arXiv"></a>
  <a href="https://github.com/modelscope/AgentEvolver"><img src="https://img.shields.io/github/stars/modelscope/AgentEvolver?style=social" alt="GitHub Stars"></a>
</p>


<!-- <p align="center">
  <strong>AgentEvolver: An Efficient Self-Evolving Agent System</strong><br>
</p> -->

**AgentEvolver** is an end-to-end, self-evolving training framework that unifies self-questioning, self-navigating, and self-attributing into a cohesive system. It empowers agents to autonomously
improve their capabilities, aiming for efficient, cost-effective, and continuous capability evolution.


## 📰 News

- **[2025-11]** 📄 [The AgentEvolver Technical Report is now available](https://arxiv.org/abs/2511.10395), detailing the framework’s architecture, methodology, and key findings.
- **[2025-11]** 🧩 AgentEvolver v1 has been released now!


## ✨ Why AgentEvolver



🧠 AgentEvolver provides three **Self-Evolving Mechanisms** from Environment to Policy:

- **Automatic Task Generation (Self-Questioning)** – Explore the environment and autonomously create diverse tasks, eliminating costly manual dataset construction.
- **Experience-guided Exploration (Self-Navigating)** – Summarize and reuse cross-task experience, guiding higher-quality rollouts and improving exploration efficiency.
- **Attribution-based Credit Assignment (Self-Attributing)** – Process long trajectories to uncover the causal contribution of intermediate steps, enabling fine-grained and efficient policy optimization.

<p align="center">
 <img src="docs/img/flowchart.png" alt="AgentEvolver Flowchart" width="80%">
</p>




## 🔧 Architecture Design
AgentEvolver adopts a service-oriented dataflow architecture, seamlessly integrating environment sandboxes, LLMs, and experience management into modular services.

<p align="center">
 <img src="docs/img/system.png" alt="system framework" width="80%">
</p>


- **Environment Compatibility** – Standardized interfaces for seamless integration with a wide range of external environments and tool APIs.
- **Flexible Context Manager** – Built-in utilities for managing multi-turn contexts and complex interaction logic, supporting diverse deployment scenarios.
- **Modular & Extensible Architecture** – Decoupled components allow easy customization, secondary development, and future algorithm upgrades.


## 🌟 Benchmark Performance

Performance comparison on the AppWorld and BFCL-v3 benchmarks. AgentEvolver achieves superior results while using substantially fewer parameters than larger baseline models.

<p align="center">
 <img src="docs/img/performance.png" alt="Benchmark Performance" width="80%">
</p>

Performance on two benchmarks. Columns show avg@8 and best@8 for each benchmark, plus their averages (Avg.). All values are in percent (%). **Bolded numbers** highlight the best results.

<table>
  <thead>
    <tr>
      <th rowspan="2" align="left"><strong>Model</strong></th>
      <th rowspan="2" align="center"><strong>Params</strong></th>
      <th colspan="2" align="center" style="text-align: center;"><strong>AppWorld</strong></th>
      <th colspan="2" align="center" style="text-align: center;"><strong>BFCL v3</strong></th>
      <th colspan="2" align="center" style="text-align: center;"><strong>Avg.</strong></th>
    </tr>
    <tr>
      <th align="center">avg@8</th>
      <th align="center">best@8</th>
      <th align="center">avg@8</th>
      <th align="center">best@8</th>
      <th align="center">avg@8</th>
      <th align="center">best@8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">Qwen2.5-7B</td>
      <td align="center">7B</td>
      <td align="center">1.8</td>
      <td align="center">5.6</td>
      <td align="center">29.8</td>
      <td align="center">42.4</td>
      <td align="center">15.8</td>
      <td align="center">24.0</td>
    </tr>
    <tr>
      <td align="left">+Questioning</td>
      <td align="center">7B</td>
      <td align="center">23.2</td>
      <td align="center">40.3</td>
      <td align="center">49.0</td>
      <td align="center">60.6</td>
      <td align="center">36.1</td>
      <td align="center">50.5</td>
    </tr>
    <tr>
      <td align="left">+Questioning&amp;Navigating</td>
      <td align="center">7B</td>
      <td align="center">26.3</td>
      <td align="center">43.1</td>
      <td align="center">53.3</td>
      <td align="center">61.0</td>
      <td align="center">39.8</td>
      <td align="center">52.1</td>
    </tr>
    <tr>
      <td align="left">+Questioning&amp;Attributing</td>
      <td align="center">7B</td>
      <td align="center">25.7</td>
      <td align="center">43.7</td>
      <td align="center">56.8</td>
      <td align="center">65.3</td>
      <td align="center">41.3</td>
      <td align="center">54.5</td>
    </tr>
    <tr>
      <td align="left"><strong>AgentEvolver (overall)</strong></td>
      <td align="center"><strong>7B</strong></td>
      <td align="center"><strong>32.4</strong></td>
      <td align="center"><strong>51.2</strong></td>
      <td align="center"><strong>57.9</strong></td>
      <td align="center"><strong>69.0</strong></td>
      <td align="center"><strong>45.2</strong></td>
      <td align="center"><strong>60.1</strong></td>
    </tr>
    <tr>
      <td colspan="8"></td>
    </tr>
    <tr>
      <td align="left">Qwen2.5-14B</td>
      <td align="center">14B</td>
      <td align="center">18.0</td>
      <td align="center">31.4</td>
      <td align="center">41.6</td>
      <td align="center">54.1</td>
      <td align="center">29.8</td>
      <td align="center">42.8</td>
    </tr>
    <tr>
      <td align="left">+Questioning</td>
      <td align="center">14B</td>
      <td align="center">44.3</td>
      <td align="center">65.5</td>
      <td align="center">60.3</td>
      <td align="center">72.1</td>
      <td align="center">52.3</td>
      <td align="center">68.8</td>
    </tr>
    <tr>
      <td align="left">+Questioning&amp;Navigating</td>
      <td align="center">14B</td>
      <td align="center">45.4</td>
      <td align="center">65.3</td>
      <td align="center">62.8</td>
      <td align="center">74.5</td>
      <td align="center">54.1</td>
      <td align="center">69.9</td>
    </tr>
    <tr>
      <td align="left">+Questioning&amp;Attributing</td>
      <td align="center">14B</td>
      <td align="center">47.8</td>
      <td align="center">65.6</td>
      <td align="center">64.9</td>
      <td align="center">76.3</td>
      <td align="center">56.4</td>
      <td align="center">71.0</td>
    </tr>
    <tr>
      <td align="left"><strong>AgentEvolver (overall)</strong></td>
      <td align="center"><strong>14B</strong></td>
      <td align="center"><strong>48.7</strong></td>
      <td align="center"><strong>69.4</strong></td>
      <td align="center"><strong>66.5</strong></td>
      <td align="center"><strong>76.7</strong></td>
      <td align="center"><strong>57.6</strong></td>
      <td align="center"><strong>73.1</strong></td>
    </tr>
  </tbody>
</table>


## 🚀 Quick Start
### Step 1. Basic Dependency Installation

Make sure you have **conda** and **cuda toolkit** installed.

Then, set up the training environment by running the script

```bash
bash install.sh
```


### Step 2. Setup Env-Service (Appworld as example)
The script below sets up an environment for appworld.

```bash
cd env_service/environments/appworld && bash setup.sh
```

### Step 3. Setup ReMe (Optional)
Set up the ReMe for experience management by running the script:
```bash
bash external/reme/install_reme.sh
```
For more detailed installation, please refer to [ReMe](https://github.com/agentscope-ai/ReMe).

### Step 4. Begin Training! 🚀 🚀
Copy the `example.env` file to `.env` and modify the parameters, including your **API key**, **conda path**.

Using AgentEvolver launcher to start environment, log dashboard and training process altogether.

```bash
conda activate agentevolver

# option 1: minimal example without ReMe (using built-in datasets within environments)
python launcher.py --conf examples/basic.yaml --with-appworld

# option 2: full example with ReMe (questioning + navigating + attributing)
python launcher.py --conf examples/overall.yaml --with-appworld --with-reme
```

## 🧩 Advanced Usage

### 🔧 Manual Execution

For users requiring fine-grained control over the training pipeline, we provide standalone execution scripts: 

- `bash examples/run_basic.sh` - Execute basic RL pipeline with GRPO using built-in datasets within environments.
- `bash examples/run_overall.sh` - Run the complete self-evolving AgentEvolver pipeline with fully customizable configurations.

Refer to the  **[QuickStart](docs/tutorial/quick_start.md)** for detailed usage instructions and configuration parameters.

### 📄 Documentation

For detailed usage and customization, please refer to the following guidelines:

- **[Environment Service](docs/guidelines/env_service.md)** - Set up and manage environment instances, integrate custom environments
- **[Task Manager](docs/guidelines/task_manager.md)** - Explore environments, generate synthetic tasks, and curate training data for agent evolution
- **[Experience Manager](docs/guidelines/exp_manager.md)** - Configure experience pool management and self-navigating mechanisms
- **[Advantage Processor](docs/guidelines/adv_processor.md)** - Implement self-attributing mechanisms with ADCA-GRPO for fine-grained credit assignment

For API documentation and more details, visit our [documentation site](docs/index.md).

## 🔮 Upcoming
- **Evolution in multi-agent scenarios** – Investigate autonomous co-evolution strategies for agents operating within shared, interactive environments.
- **Cross-stage collaborative self-evolution** – Explore methods that couple questioning, navigating, and attributing into coordinated loops for mutual enhancement.

<!-- ## 🌟 Contact Us -->

## 🙏 Acknowledgements
This project builds upon the excellent work of several open-source projects:

- [ReMe](https://github.com/agentscope-ai/ReMe) - for experience summarization and management;
- [veRL](https://github.com/volcengine/verl) - for distributed RL training;
- [mkdocs](https://github.com/mkdocs/mkdocs) - for documentation.

## 📚 Citation
If you find this work useful, please consider citing:

```bibtex
@misc{AgentEvolver2025,
  title         = {AgentEvolver: Towards Efficient Self-Evolving Agent System},
  author        = {Yunpeng Zhai and Shuchang Tao and Cheng Chen and Anni Zou and Ziqian Chen and Qingxu Fu and Shinji Mai and Li Yu and Jiaji Deng and Zouying Cao and Zhaoyang Liu and Bolin Ding and Jingren Zhou},
  year          = {2025},
  eprint        = {2511.10395},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2511.10395}
}
```