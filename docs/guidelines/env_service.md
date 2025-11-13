# EnvService Overview

To evolve agents in a data-driven manner within dynamic environments, a robust and scalable environment management layer is essential. **EnvService** fulfills this role by providing a unified service that manages the **lifecycle of training environment instances**, enabling seamless interaction, evaluation, and control during agent training.

**EnvService** is designed to operate independently of other components and can be launched and used in isolation. It exposes a well-structured API and integrates with distributed execution frameworks like **Ray**, enabling support for large-scale agent evolution across heterogeneous environments.

## Core Responsibilities of EnvService

- **Dynamic Environment Instantiation**  
  Dynamically create environment instances based on task specifications, parameters, and environment types (e.g., `appworld`, `bfcl`,`webshop`).

- **Lifecycle and Resource Management**  
  Track environment usage and automatically release inactive instances to ensure efficient resource utilization.

- **Unified Interaction Interface**  
  Provide a RESTful API for interacting with environments — including initialization, stepping, evaluation, and instance cleanup.

- **Modular Environment Loading**  
  Support dynamic import and registration of custom environment modules without modifying core service logic.

- **Asynchronous and Distributed Execution**  
  Leverage **Ray actors** to run each environment instance independently, enabling scalable and concurrent execution.

EnvService serves as the **execution backbone** for the agent evolution pipeline, enabling consistent interaction patterns, efficient resource handling, and flexible integration with various synthetic or user-defined environments.

---
## Environment Setup

We provide a simple setup script and a `Dockerfile` to help you get started quickly.  
In this subsection, we explain the environment installation process.  
The launch instructions are provided in the next section.  

More environments will be supported soon.

### Appworld

To install the `appworld` environment, navigate to the following directory and run the setup script. This script will configure environment variables, create a Conda environment with Python 3.11, install necessary dependencies, initialize Appworld, and download required data:

```bash
cd env_service/environments/appworld
bash setup.sh
```

### BFCL
To install the `BFCL` environment, navigate to the following directory and run the setup script. This script will configure environment variables, create a Conda environment with Python 3.11, install necessary dependencies, initialize BFCL, and download required data:

```bash
cd env_service/environments/bfcl
bash setup.sh
```


--- 
## Launch Environment Service 
After installation, you can launch Environment Service with the following commands:

### Appworld

To launch the `appworld` environment service, run the script in the `env_service/launch_script/appworld.sh`:

```bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate appworld
cd env_service/launch_script
bash appworld.sh
```

By default, the service will start on `127.0.0.1:8080`.  
If you need to change the host (`--portal`) or port (`--port`), edit the following line inside `env_service/launch_script/appworld.sh`:

```bash
exec python -m env_service.env_service --env appworld --portal 127.0.0.1 --port 8080
```

Feel free to replace the IP address or port number as needed to suit your environment.


### BFCL

To launch the `bfcl` environment service, run the script in the `env_service/launch_script/bfcl.sh`:

```bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bfcl
cd env_service/launch_script
bash bfcl.sh
```

By default, the service will start on `127.0.0.1:8080`.  
If you need to change the host (`--portal`) or port (`--port`), edit the following line inside `env_service/launch_script/bfcl.sh`:

```bash
exec python -m env_service.env_service --env bfcl --portal 127.0.0.1 --port 8080
```

Feel free to replace the IP address or port number as needed to suit your environment.



---
## Environment Service Interface 

You can also launch the environment service manually without using a script by running:

```bash
python -m env_service.env_service [--args]
```

Here are the available command-line arguments:

| Argument         | Type    | Default      | Description                                                                                                                             |
|------------------|---------|--------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| `--env`          | `str`   | `"appworld"` | The name of the environment to run.                                                                                                     |
| `--env_file_name`| `str`   | `None`       | Optional. Specific file name of the environment module. Defaults to `"{env}_env.py"`.                                                   |
| `--portal`       | `str`   | `"0.0.0.0"`  | The IP address to bind the server to. Use `"127.0.0.1"` for local only.                                                                 |
| `--port`         | `int`   | `8000`       | Port number to run the server on.                                                                                                       |
| `--debug`        | `bool`  | `False`      | Whether to run the server in debug mode (`True`) or production mode (`False`), where useless logger will be removed in production mode. |

### Example

Start the server for `appworld` on port `8080` and localhost:

```bash
python -m env_service.env_service --env appworld --portal 127.0.0.1 --port 8080 --debug True
```

This gives you more flexibility to control and integrate the environment service in your own workflows.



## API Interaction Flow
Our EnvService communicates via HTTP requests, primarily through the `EnvClient` located at `env_service/env_client.py`, which handles the service connection and has already been integrated into **AgentEvolver**.

1. **Query Task Candidates (Optional):**  
   `POST /get_env_profile`  
   → Returns a list of available `task_id`s for a given environment type.

2. **Create Instance:**  
   `POST /create`  
   → Initializes a new environment with `env_type`, `task_id`, and custom parameters.  
   → Returns the initial state.

3. **Interact with Environment:**  
   `POST /step`  
   → Sends an action to the environment instance.  
   → Returns the next state, reward, and done signal.

4. **Evaluate Agent Behavior (Optional):**  
   `POST /evaluate`  
   → Used for scoring overall task performance (e.g., episodic evaluation).

5. **Get Instance Information (Optional):**  
   `POST /get_info`  
   → Retrieve metadata or progress information from the environment.

6. **Release Instance:**  
   `POST /release`  
   → Frees the environment resources explicitly.

7. **Health Check:**  
   `GET /healthz`  
   → Basic service status check.



##  Input & Output Data Structure

To help users interact effectively with the **Appworld** environment through the EnvService API, this section details the expected input and output data structures used in the `/create` and `/step` calls.

These examples illustrate what kind of requests to send and how to interpret the returned results.
More examples can be found in `env_service/interface.md`
---

### 🛠️ `/create` – Environment Initialization

**Description:** Initializes a new environment instance and prepares its initial state.

#### 🔽 Input Parameters

```json
{
  "env": "appworld",
  "instance_id": "my-instance-id-001",
  "task_id": "task-001",
  "params": {
    "simple": false,
    "prompt": true
  }
}
```

| Field         | Type     | Description                                                                 |
|--------------|----------|-----------------------------------------------------------------------------|
| `env`        | `string` | The environment to use. Set to `"appworld"`.                                |
| `instance_id`| `string` | A user-defined identifier for tracking the environment instance.            |
| `task_id`    | `string` | The task assigned to this instance. Tasks can differ by difficulty or type. |
| `params`     | `dict`   | Additional configuration options:                                           |
|              | `simple` | `bool` – If true, uses a simplified interaction format.                      |
|              | `prompt` | `bool` – If true, includes the environment prompt template in the state.     |

#### ⏎ Output (Initial State)

```json
{
  "state": [
    {
      "role": "system",
      "content": "[Structured prompt with task description and tool info]"
    },
    {
      "role": "user",
      "content": "[Task instruction from Appworld]"
    }
  ],
  "info": {
    "instance_id": "my-instance-id-001",
    "task_id": "task-001"
  }
}
```

---

### 🔄 `/step` – Sending an Action to the Environment

**Description:** Advances the environment one step using a user-generated action.

#### 🔽 Input Format

```json
{
  "action": {
    "role": "assistant",
    "content": "```python\nopen_app(\"Calendar\")\n```",
    "tool_calls": []
  }
}
```

| Field        | Type   | Description                                                                 |
|-------------|--------|-----------------------------------------------------------------------------|
| `role`      | `str`  | Typically `"assistant"`, representing the agent performing the action.      |
| `content`   | `str`  | The agent's natural language or code action. May include Python code blocks.|
| `tool_calls`| `list` | Optional list of structured tool calls (used for advanced interactions).     |

> ✅ You can omit `tool_calls` if your action is just a plain code string.

---

### 📤 Response from `/step`

```json
{
  "state": [
    {
      "role": "assistant",
      "content": "Output:\n```\nApp opened successfully\n```"
    }
  ],
  "reward": 1.0,
  "is_terminated": true,
  "info": {}
}
```

| Field           | Type     | Description                                                               |
|----------------|----------|---------------------------------------------------------------------------|
| `state`        | `list`   | Environment’s response, typically including tool output or result summary. |
| `reward`       | `float`  | Task performance score. 1.0 = success, 0.0 = failure (configurable).       |
| `is_terminated`| `bool`   | Indicates whether the task is completed.                                  |
| `info`         | `dict`   | Reserved for additional metadata (currently empty).                        |

---

### 💡 Tips for Use

- The `content` field should contain a well-formatted Python action or a command interpretable by Appworld.
- You may use Markdown-style code blocks (` ```python `) to wrap executable code for better readability.
- If `is_terminated: true`, the environment expects no further actions.
- Use the `reward` to evaluate success in training or evaluation settings.
- The `simple` param in `/create` makes the prompt and output shorter — useful for debugging or fast testing.

---

For more details on tool calls, prompt generation, or evaluation metrics, refer to the **Appworld Environment Internals** section.
"""
