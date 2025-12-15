# 🎮 AgentEvolver Game Arena

AgentEvolver Game Arena is a platform for **training, evaluating, and observing AI agents in multi-agent social board games**.  
It focuses on games with **hidden information, negotiation, deception, and cooperation**, enabling users to study and compare agent behavior in complex social decision-making environments.

The platform supports **web-based gameplay and observation**, **batch model evaluation**, and **agent training**, making it suitable for researchers, model evaluators, and AI enthusiasts.

---

## ✨ What Can You Do?

### 1. Watch or Play AI Agents in Your Browser

A web-based interface allows you to:

- **Observer Mode** – Watch AI agents play in real time and inspect their reasoning, communication, and strategic decisions  
- **Participate Mode** – Join a game as a human player and play alongside or against AI agents to test their interactive capabilities  

**Supported Games**

We currently support two games, Avalon and Diplomacy, both of which are strategy games involving long-term reasoning.

<div align="center">
  <figure style="display: inline-block; margin: 0 1em; vertical-align: top;">
    <img src="../docs/img/games/avalon_demo_extracted.gif" alt="Avalon Demo" width="45%" style="display: block;" />
    <figcaption style="text-align: center; color: #666;">Avalon</figcaption>
  </figure>
  <figure style="display: inline-block; margin: 0 1em; vertical-align: top;">
    <img src="../docs/img/games/diplomacy_demo_extracted.gif" alt="Diplomacy Demo" width="45%" style="display: block;" />
    <figcaption style="text-align: center; color: #666;">Diplomacy</figcaption>
  </figure>
</div>

---

### 2. Evaluate AI Models at Scale

AgentEvolver provides a built-in evaluation framework to **systematically compare model performance**:

- Run multiple games in parallel for statistically meaningful results  
- Control game settings and model assignments via configuration files  

> 📊 *Example evaluation summaries and visualized results will be added here.*

---

### 3. Train AI Agents

AgentEvolver is designed to support **end-to-end training of AI agents in social board games**, enabling agents to learn from interaction, feedback, and long-horizon outcomes.

- Training agents directly within game environments  
- Support for reinforcement learning–based methods (e.g., GRPO)  

> 📈 *Training curves, learning dynamics, and performance evolution plots will be added here.*

---

## 🚀 Getting Started

### Install Dependencies

Run:

    pip install -r requirements.txt

(Optional) Set environment variables:

    export OPENAI_API_KEY=your_api_key
    export MODEL_NAME=your_default_model

---

### Launch the Web Interface (Recommended)

Start the server:

    python games/web/server.py

Then open your browser at:

    http://localhost:8000

From the web interface you can:

1. Select a game (Avalon or Diplomacy)  
2. Choose a mode (Observer or Participate)  
3. Configure players and models  
4. Start the game  

---

### Run a Model Evaluation

Example command:

    python games/evaluation/run_eval.py \
        --game avalon \
        --config games/games/avalon/configs/task_config.yaml \
        --num-games 10

After completion, a summary similar to the following will be displayed:

    Evaluation Results Summary – AVALON
    total_games: 10
    successful_games: 10
    good_victory_mean: 0.6000
    ...

---

## ⚙️ Configuration Overview

All games and evaluations are controlled via **YAML configuration files**, allowing you to adjust behavior without modifying code.

Example structure:

    game:
      name: avalon
      num_players: 5
      language: en

    default_model:
      model_name: qwen-plus
      temperature: 0.7

    roles:
      assassin:
        model_name: custom-model

With configuration files you can:

- Assign different models to different roles  
- Adjust model parameters and game settings  
- Reproduce experiments consistently  

---

## 📚 Additional Documentation

- Web interface details: games/web/README.md  
- Training and algorithm details: see the main project documentation  

---

## 📄 License

Please refer to the main project LICENSE file.
