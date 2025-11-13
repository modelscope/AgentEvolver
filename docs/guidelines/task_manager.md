To evolve an agent in a data-driven manner within your own environment, the first step is to collect training data that maps the agent's abilities to your requirements.

Task Manager provides the training data for AgentEvolver. It is responsible for:

- Exploring unknown environments and profiling potential tasks,  
- Discovering new synthetic tasks and capturing user requirements,  
- Curating training tasks and managing their quality and quantity,  
- Providing a built-in synthetic reward mechanism as a fallback.  

In this section, we introduce Task Manager and explain how to efficiently collect appropriate training data for agent training, including the configuration of Environment Profiling, Task Derivation, and Task Curation strategies.

## Collect Your First Training Data

To collect your training data, perform the following five steps:

1. Integrate your environment with Environment Service (as described in the previous section).  
2. Profile the environment.  
3. Configure the strategies to be applied, including their parameters.  
4. Execute Task Manager and collect the training data.
5. Check your data.

### 1. Adopt the Environment

Assume the environment has already been integrated with Environment Service. If not, please refer to the previous section for setup instructions.

### 2. Profile the Environment

Task Manager requires not only the API specifications but also conceptual knowledge of the environment. For example, in a file system, APIs define file operations, but they do not capture high-level concepts such as file types or formats. These conceptual elements must be explicitly represented.

We introduce the **Environment Profile** to capture these concepts. An Environment Profile is a JSON file that specifies *Entity*, *Attribute*, and *Operation* definitions for the environment.

- **Entity**: Represents a core object in the environment. Entities are the targets of interaction and can typically be created, modified, or deleted.  
- **Attribute**: Defines descriptive properties or metadata of an entity. Attributes provide contextual information but are not executable actions.  
- **Operation**: Specifies the actions that can be performed on an entity. Operations represent the functional capabilities of the environment and are often aligned with API calls.  

Additionally, the Environment Profile defines **task preferences**, which control the style and scope of the generated tasks.

A basic Environment Profile example is shown below:

```json
{
  "name": "Alice",
  "background": "A general user working with a file system.",
  "entities": [
    {
      "name": "file",
      "description": "A file in a file system.",
      "attrs": {
        "name": "The name of the file.",
        "size": "The size of the file in bytes.",
        "type": "The type of the file, e.g. text, image, video, etc.",
        "parent": "The parent directory of the file."
      },
      "opts": [
        { "name": "create", "description": "Create a new file." },
        { "name": "delete", "description": "Delete a file." },
        { "name": "read", "description": "Read a file." },
        { "name": "write", "description": "Write to a file." }
      ]
    },
    {
      "name": "directory",
      "description": "A directory in a file system.",
      "attrs": {
        "name": "The name of the directory.",
        "parent": "The parent directory of the directory."
      },
      "opts": [
        { "name": "create", "description": "Create a new directory." },
        { "name": "delete", "description": "Delete a directory." },
        { "name": "list", "description": "List the contents of a directory." }
      ]
    }
  ],
  "task_preference": {
    "num_entities": 2,
    "num_opts": 3,
    "relation_difficulty": 3
  }
}
```

In this profile, entities `file` and `directory` are defined with attributes (`name`, `size`, `type`, `parent`) and operations (`create`, `delete`, `read`, `write`, `list`). Based on these definitions, Task Manager gains a structured understanding of the environment to support task derivation and curation.

To create your own Environment Profile, copy the template `environment_profile_template.json` to `environment_profile.json` and fill in the details. Using an LLM to assist in drafting the profile can reduce manual effort.

### 3. Configure the Strategies

Transforming profiles into synthetic tasks involves two stages: **task derivation** and **task curation**.

**Task derivation** is the process of generating candidate tasks from the profile. During derivation, exploration and summarization are performed under the guidance of a chosen strategy. Strategies determine how the environment is traversed and how structured tasks are extracted from exploration trajectories.

**Task curation** ensures task quality and diversity. Filters are applied to discard infeasible, redundant, or irrelevant tasks. Mixture strategies combine tasks from multiple sources and control properties such as difficulty distribution.

By default, Task Manager provides:

* *RandomWalk Strategy* for task derivation,
* *DeduplicationFilter*, *FeasibilityFilter*, and *UnifiedMixtureStrategy* for task curation.

These can be configured in the YAML configuration file. After configuring strategies, the whole configuration file looks like this:

```yaml
task_manager:
  # tasks will be explored once if set. use it if you want to keep the same explorations.
  train_data_path: tasks_explored.train.json
  val_data_path: tasks_explored.val.json
  # model used to explore the environment
  llm_client: qwen-plus
  # repetation of exploration
  n: 0
  # env profiles used in exploration
  env_profile: cookbook/env_profiles/appworld.json
  # batch size of dynamic synthetic data
  bs: ${data.train_batch_size} # use the same batch size as train_batch_size
  # number of threads for exploration
  num_explore_threads: 16

  # mixture strategy
  mixture:
    # whether to use original tasks provided by the environment
    use_original_tasks: True
    synthetic_data_ratio: 0.0
    # whether to shuffle tasks *after* mixture
    shuffle: True

  # the grader used to evaluate tasks
  grader:
    # grader used to evaluate tasks: env, llm
    original_grader: env
    synthetic_grader: llm

  # strategy used to explore the environment
  strategy: random
  strategy_args:
    max_explore_step: 30
    max_llm_retries: 6
    env_url: ${env_service.env_url} # refer to env_service.env_url
    exploration_llm_temperature: 1.0
    exploration_llm_top_p: 1.0
    exploration_llm_top_k: 100
```

### 4. Start Task Synthesis

Once configuration is complete, task synthesis can be initiated.

1. Start the Environment Service.
2. Start Task Manager.

#### Standalone Mode

Task Manager can be executed in standalone mode for simple task synthesis.

Example command:

```bash
$ python -m agentevolver.module.task_manager
```

The synthesis progress will be displayed. When the process completes, the path to the generated tasks will be printed.

#### Integrated Mode

In most workflows, Task Manager is integrated with AgentEvolver. Launching AgentEvolver automatically starts the training and task synthesis pipeline.

!!! info "Standalone vs Integrated"
    Task Manager can be run independently for rapid prototyping or small-scale data generation. It is recommended to tune strategies in standalone mode, and then use integrated mode in production, where additional features are available within AgentEvolver.

### 5. Check the Data

The generated synthetic tasks are stored in the path specified in the YAML configuration file.

```yaml
task_manager:
  # where to save the generated tasks
  train_data_path: tasks_explored.train.json # tasks will be explored once if set. use it if you want to keep the same explorations.
  val_data_path: tasks_explored.val.json # tasks will be explored once if set. use it if you want to keep the same explorations.
```

!!! note "Dynamic vs Static"
    `train_data_path` and `val_data_path` set, tasks will be explored once and saved to the specified path. If no path is set for integrated mode, Task Manager will generate tasks dynamically during training. All synthetic tasks will be discarded after training.

Inspect the generated data to ensure it aligns with your training requirements.


## Workflow of Task Manager

In data-driven model optimization, agent training is formulated as trajectory tuning over environment-specific tasks. Consequently, the quality of training data directly determines the resulting agent capabilities. However, in real environments, acquiring and controlling the quality of training tasks is inherently difficult.

Task Manager addresses this challenge by providing a dynamic and general-purpose workflow for environment exploration, task generation, and quality control.

![The Whole Workflow of Task Manager](../img/task_manager/workflow.png)

From the figure, the workflow consists of three major steps and corresponding components:

1. **Environment Exploration**
    - **Environment Service** – Provides interface for the environment.
    - **Environment Profile** – Describes the environment.
2. **Task Derivation**:
    - **Strategy** – Control the exploration and summarization process.
3. **Task Curation**:
    - **Filter** – Control the quality of tasks.
    - **Mixture Strategy** – Control the distribution of tasks.

And

- **Judge**: Provide rewards for training.

The following sections describe each component of Task Manager in detail, including extension points for customization.

## Environment Profiling

An Environment Profile describes the concepts of an environment using **entities**, **attributes**, and **operations**. Similar to object-oriented programming and database schemas, these components are considered fundamental.

* **Entity**: Represents an object in the environment.
* **Attributes**: Define the properties of the entity.
* **Operations**: Specify the actions that can be applied to the entity.

For example:

```
Entity: File
Attributes
    - name: The name of the file.
    - size: The size of the file in bytes.
    - type: The type of the file, e.g. text, image, video, etc.
    - permission: The permission of the file.
Operations
    - create: Create a new file.
    - delete: Delete a file.
    - read: Read a file.
    - write: Write to a file.
    - chmod: Change the permission of a file.
```

The granularity of a profile is flexible. With the assistance of LLMs, profiles can be constructed at multiple levels, ranging from a single generic entity to highly specialized entities. The choice of granularity is a trade-off between manual specification and the capability of the LLM to generalize.

Task Manager leverages the Environment Profile to recognize concepts, explore relationships between entities, and synthesize meaningful tasks. Operations are combined to form candidate solutions reflecting real-world problem-solving.

Users may optionally specify a **User Preference** in addition to the Environment Profile. Preferences define expectations for the agent's capabilities, such as desired task difficulty or task categories.

### Write a Profile

Profiles can be specified in **JSON** (recommended) or in **Python**.

Top-level structure:

```json
{
  "name": string,
  "background": string,
  "entities": [ ... ],
  "task_preference": {
    "num_entities": integer,
    "num_opts": integer,
    "relation_difficulty": integer
  }
}
```

Example entity definition:

```json
{
  "name": "file",
  "description": "A file in a file system.",
  "attrs": {
    "name": "The name of the file."
  },
  "opts": [
    { "name": "create", "description": "Create a new file." }
  ]
}
```

A minimal working example:

```json
{
  "name": "Alice",
  "background": "A general user working with a file system.",
  "entities": [
    {
      "name": "file",
      "description": "A file in a file system.",
      "attrs": {
        "name": "The name of the file.",
        "size": "The size of the file in bytes.",
        "type": "The type of the file (e.g., text, image, video).",
        "parent": "The parent directory of the file."
      },
      "opts": [
        { "name": "create", "description": "Create a new file." },
        { "name": "delete", "description": "Delete a file." },
        { "name": "read", "description": "Read a file." },
        { "name": "write", "description": "Write to a file." }
      ]
    },
    {
      "name": "directory",
      "description": "A directory in a file system.",
      "attrs": {
        "name": "The name of the directory.",
        "parent": "The parent directory of the directory."
      },
      "opts": [
        { "name": "create", "description": "Create a new directory." },
        { "name": "delete", "description": "Delete a directory." },
        { "name": "list", "description": "List the contents of a directory." }
      ]
    }
  ],
  "task_preference": {
    "num_entities": 2,
    "num_opts": 3,
    "relation_difficulty": 3
  }
}
```
For examples in real environments, please refer to `cookbook` in the root directory.

If Python is preferred, an example is

```python
from agentevolver.module.task_manager.env_profiles import EnvEntity, EnvEntityOpt, TaskPreference, EnvProfile

spotify = EnvEntity(
    name="Spotify",
    description="A music streaming service with song, album, and playlist management.",
    attrs={
        "Song Library": "Songs saved by the user.",
        "Album Library": "Albums saved by the user.",
        "Playlists": "User-created or followed playlists."
    },
    opts=[
        EnvEntityOpt("play_song", "Play a song, album, or playlist."),
        EnvEntityOpt("like_song", "Like songs or albums."),
        EnvEntityOpt("unfollow_artist", "Unfollow an artist."),
        EnvEntityOpt("follow_artist", "Follow an artist."),
        EnvEntityOpt("create_playlist", "Create a new playlist."),
        EnvEntityOpt("remove_song", "Remove songs from library or playlist."),
        EnvEntityOpt("export_library", "Export song/album/playlist data.")
    ]
)

gmail = EnvEntity(
    name="Gmail",
    description="An email service for sending, receiving, labeling, and managing emails.",
    attrs={
        "Inbox": "List of received email threads.",
        "Outbox": "List of sent email threads.",
        "Labels": "Custom labels to organize emails."
    },
    opts=[
        EnvEntityOpt("send_email", "Send an email."),
        EnvEntityOpt("forward_email", "Forward an email."),
        EnvEntityOpt("reply_email", "Reply to an email."),
        EnvEntityOpt("delete_email", "Delete emails."),
        EnvEntityOpt("label_email", "Label emails."),
        EnvEntityOpt("star_email", "Star or unstar email threads.")
    ]
)

env_profile = EnvProfile(
    name="Bob",
    background="A general computer user.",
    task=TaskPreference(
        num_entities=2,
        num_opts=3,
        relation_difficulty=3,
    )
)

env_profile.reg_entities([spotify, gmail])
```

More examples can be found in `agentevolver/module/task_manager/prelude_profiles`.

## Task Derivation

Task Derivation is the initial stage of synthetic task generation. It transforms the Environment Profile into preliminary task drafts by applying exploration and synthesis strategies.

The primary objectives are:

1. **Exploration** – Cover the environment systematically or stochastically.
2. **Summarization** – Convert exploration trajectories into concise, structured candidate tasks.

Available strategies:

* **RandomWalk Strategy** – Random exploration producing a diverse task set.
* Additional strategies will be introduced in future versions.

### RandomWalk Strategy

The RandomWalk Strategy is simple yet effective. It explores the environment by sampling entities and operations at random, generating diverse trajectories that can be summarized into tasks.

Parameters:

```yaml
task_manager:
  strategy: random
  strategy_args:
    max_explore_step: 30
    max_llm_retries: 6
    env_url: ${env_service.env_url} # refer to env_service.env_url
    exploration_llm_temperature: 1.0
    exploration_llm_top_p: 1.0
    exploration_llm_top_k: 100
```

## Task Curation

Task Curation ensures the quality and diversity of tasks generated during derivation by applying filters and mixture strategies.

* **Filters**
    * *DeduplicationFilter*: Removes redundant or near-duplicate tasks.
    * *FeasibilityFilter*: Removes tasks that cannot be executed in the environment.

* **Mixture Strategies**
    * *UnifiedMixtureStrategy*: Combines tasks from multiple sources to maintain balance.

Goals of curation:

* **Quality assurance** – Ensure valid, feasible, and logically sound tasks.
* **Diversity preservation** – Avoid bias toward a single task type.
* **Dynamic control** – Adjust task selection according to agent progress.

### DeduplicationFilter

Removes duplicate or highly similar tasks to improve data diversity. Enabled by default.

### FeasibilityFilter

Filters out tasks that cannot be completed given the environment constraints.

## Synthetic Reward

Task Manager provides a built-in **synthetic reward** as a fallback, enabling training without requiring user-defined reward functions.

Key properties:

* **Generality** – Applicable across diverse environments.
* **Zero-configuration** – Works out of the box.
* **Extensibility** – Can be replaced or extended with custom reward functions.

Typical reward components:

* **Relevance check** – Whether the trajectory matches the task.
* **Success check** – Whether the task is successfully completed.
* **Efficiency check** – Whether the task is solved within reasonable steps.

> Although the built-in synthetic reward provides a functional fallback, we strongly recommend implementing domain-specific reward functions to achieve optimal agent performance.

Configuration example:

```yaml
task_manager:
  grader:
    original_grader: env
    synthetic_grader: llm
```

## Extend Task Manager

Task Manager is designed as a **modular and extensible framework**, adaptable to different training scenarios. To extend Task Manager, users can implement and replace the components used in three stages in the pipeline, including **Strategy**, **Filter**, **Mixture Strategy**, and **Judge**.