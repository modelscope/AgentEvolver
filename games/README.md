<p align="center">
 <img src="../docs/img/games/game_logo.jpeg" alt="AgentEvolver Logo" width="90%">
</p>


# 🎮 AgentEvolver Game Arena

**A unified arena for interaction, evaluation, and training of AI agents in social reasoning games.**


AgentEvolver Game Arena extends **AgentEvolver** into multi-agent social game environments. By focusing on board games with **multi-round, long-horizon interaction and clear reward rules**, and rich strategic spaces involving **hidden information, negotiation and deception**, it provides a controlled setting for developing **social and strategic capabilities** beyond task execution and tool use.



## ✨ What Can You Do?

### 1. Watch or Play AI Agents in Your Browser

A web-based interface allows you to:

- **Observer Mode** – Watch AI agents play in real time and inspect their reasoning, communication, and strategic decisions  
- **Participate Mode** – Join a game as a human player and play alongside or against AI agents to test their interactive capabilities  



We currently support two games, Avalon and Diplomacy, both of which are strategy games involving long-term reasoning.

You first need to select a game, then choose your agents, configure the settings, and finally start the arena.

<table style="border: none; border-collapse: collapse;">
<tr>
<td align="center" width="50%" style="border: none; text-align: center;">
  <img src="../docs/img/games/avalon_demo_extracted.gif" alt="Avalon Demo" width="100%" />
  <br><strong>Avalon</strong>
</td>
<td align="center" width="50%" style="border: none; text-align: center;">
  <img src="../docs/img/games/diplomacy_demo_extracted.gif" alt="Diplomacy Demo" width="100%" />
  <br><strong>Diplomacy</strong>
</td>
</tr>
</table>

---

### 2. Evaluate AI Models at Scale

AgentEvolver provides a built-in evaluation framework to **systematically compare model performance**:

- Run multiple games in parallel for statistically meaningful results  
- Control game settings and model assignments via configuration files  

---

### 3. Train AI Agents

AgentEvolver is designed to support **end-to-end training of AI agents in social board games**, enabling agents to learn from interaction, feedback, and long-horizon outcomes.

- Training agents directly within game environments  
- Support for reinforcement learning–based methods (e.g., GRPO)  

> 📈 *Training curves, learning dynamics, and performance evolution plots will be added here.*

---

## 🚀 Getting Started

### Install Dependencies

We provide a minimal requirements for non-training usage:

    pip install -r games/requirements_game.txt

(Optional) Set environment variables for using LLM APIs:

    export OPENAI_BASE_URL=your_api_url
    export OPENAI_API_KEY=your_api_key


---

### Launch the Web Interface

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

To use local models, see introduction in games/evaluation/run_eval.py.

After completion, a summary similar to the following will be displayed:

    Evaluation Results Summary – AVALON
    total_games: 10
    successful_games: 10
    good_victory_mean: 0.6000
    ...

---

### Train an LLM Agent

**Prerequisites:** Before training, install the required packages using `install.sh` from the project root, and additionally install game-specific dependencies:
```bash
bash install.sh
pip install -r games/requirements_game.txt
```

Training consists of two main steps:

#### Step 1: Generate Training Tasks

Generate training task Parquet files from game-specific training configurations:

```bash
# Generate training tasks for Avalon
python games/generate_train_parquet.py \
    --game avalon \
    --config games/games/avalon/configs/train_config.yaml \
    --output ./train_avalon_tasks.parquet \
    --num_tasks 10
```

The config specifies task details and which roles/models are trainable (set `trainable: true`).


#### Step 2: Start Training

Run the training script with the generated Parquet file:

```bash
# Example for Avalon (see examples/game/avalon/run_train.sh for full example)
python -m agentevolver.main_ppo \
    --config-path="examples/game/avalon" \
    --config-name='config' \
    data.train_files="./train_avalon_tasks.parquet" \
    data.val_files="./train_avalon_tasks.parquet" \
    # ... other training parameters
```



## ⚙️ Configuration

Games and evaluations are controlled via **YAML configuration files**. Configuration structure:

- **Game settings** (`game`) – Game-specific parameters (e.g., `num_players`, `language`)
- **Model configuration** – Priority order:
  1. **Role-specific settings** (`roles` section) – Each role uses its own configuration if specified
  2. **Default model settings** (`default_model`) – Used as fallback when a role's configuration is missing

Example:

    game:
      name: avalon
      num_players: 5
      language: en
    
    default_model:
      model_name: qwen-plus
      temperature: 0.7
    
    roles:
      assassin:
        model_name: custom-model  # assassin uses custom-model, others use qwen-plus  




## 🧠 Build Your Own Agents

The AgentEvolver Game Arena is designed to be extensible and customizable. You can:

- **Develop custom agents** - Implement your own agent logic, strategies, and reasoning capabilities
- **Design memory systems** - Build memory architectures that help agents remember game history, player behaviors, and strategic patterns
- **Train models** - Use the provided training pipeline to fine-tune models for specific roles, strategies, or game scenarios

Now, try anything you want. Build your own agents, memories, or models. And one day, let's see them meet — and compete — in the arena.




## 🙏 Acknowledgments

We would like to thank the following projects and communities:

- **[AgentScope](https://github.com/modelscope/agentscope)** - For providing the multi-agent framework and infrastructure that powers our agent interactions and evaluations.

- **[Avalon-LLM](https://github.com/jonathanmli/Avalon-LLM)** - For providing the Avalon game engine and state transition logic that form the foundation of our Avalon game implementation.

- **[Diplomacy](https://github.com/diplomacy/diplomacy)** - For providing the Diplomacy game engine and state transition logic, as well as visualization and map-rendering capabilities that form the foundation of our game environment and evaluation setup.

- **[AI_Diplomacy](https://github.com/GoodStartLabs/AI_Diplomacy)** - For providing detailed, well-designed, and practical English prompts that greatly supported the development of our Diplomacy agents and training pipeline.

---

## 📄 License

Please refer to the main project LICENSE file.