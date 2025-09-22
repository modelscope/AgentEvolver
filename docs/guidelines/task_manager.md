To evolve a agent in your own environment data-driven, the first step is to collect the training data that maps the agent abilities which suit your needs.

Task Manager provides the training data for AgentEvolver. It is responsible for

- Exploring an unknown environment, profiling the potential tasks,
- Discovering new synthetic tasks, understanding the requirements of users,
- Curating the training tasks, scheduling the quality and quantity.
- Providing built-in synthetic reward as fallback.

In this section, we will introduce the Task Manager and show you how to efficiently collect proper training data for your agent training, as well as the detailed configuration of Environment Profiling, Task Derivation, and Curation strategies.

## Collect Your First Training Data

To collect your training data, there are 4 things to do:

1. Adopt your environment to Environment Service, which is supposed to be done in the previous section.
2. Profile the environment.
3. Configure the strategies to be used, as well as their parameters.
4. Run the Task Manager and collect the training data.


### 1. Adopt the environment

Suppose you have already adopted your environment to Environment Service. If not, please refer to the previous section.

### 2. Profile the Environment

Task Manager needs not only the API documents, but also the concepts in the environment. For example, in a file system, APIs provide the file operations, the types of files and their formats, however, cannot be described in the API documents, which is the concepts we call in the environment.

We introduce Environment Profile to collect the concepts. A environment profile is a JSON file that describes the *Entity*, *Attribute*, and *Operation* in the environment.

- Entity: An entity represents a fundamental object in the environment. It is the target of interactions and can usually be created, modified, or deleted.
- Attribute: Attributes describe the properties or metadata of an entity. They provide additional information that defines the state or identity of the entity but are not themselves executable actions.
- Operation: Operations define the actions that can be applied to an entity. They represent the functional capabilities available in the environment and are often aligned with API calls.

Besides, we also define the task preference in the environment profile, which is used to control the style of the tasks.

A basic environment profile is shown below.

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
        "parent": "The parent directory of the file.",
      },
      "opts": [
        {
          "name": "create",
          "description": "Create a new file."
        },
        {
          "name": "delete",
          "description": "Delete a file."
        },
        {
          "name": "read",
          "description": "Read a file."
        },
        {
          "name": "write",
          "description": "Write to a file."
        },
      ]
    },
    {
      "name": "directory",
      "description": "A directory in a file system.",
      "attrs": {
        "name": "The name of the directory.",
        "parent": "The parent directory of the directory.",
      },
      "opts": [
        {
          "name": "create",
          "description": "Create a new directory."
        },
        {
          "name": "delete",
          "description": "Delete a directory."
        },
        {
          "name": "list",
          "description": "List the contents of a directory."
        }
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

In this profile, we define entities `file` and `folder`, and their attributes `name`, `size`, `type`, `parent`, with operations `create`, `delete`, `read`, `write`, `list`. Based on these definitions, Task Manager gets a brief idea of the environment to support the task derivation and curation.

To write your own environment profile, copy the template `environment_profile_template.json` to `environment_profile.json` and fill in the details. A good idea is to use LLM to assist you in filling in the details.

### 3. Configure the Strategies

The process of making profiles into synthetic tasks includes two major steps: task derivation and task curation.

**Task derivation** is the process of generating synthetic tasks from the profile. During the derivation, environment exploration and task summarization are performed under the control of a strategy. Besides the built-in scheduler, the strategy plays an important role in prioring the path of walk in the environment and summarizing the styled tasks from the trajectories.

**Task curation** controls the quality and quantity of synthetic tasks with filters and mixture strategies. Filters are used to filter out the tasks that are not suitable for your agent, e.g. tasks that are infeasible, too complex/easy, or just does not fit your needs. Mixture strategies are used to mix the tasks from different sources, and control the characteristic (such as difficulty) of the tasks.


We implement RandomWalk strategy for task derivation, and DeduplicationFilter, FeasibilityFilter, UnifiedMixtureStrategy for task curation by default. You can find them in the config file `TODO`. More strategies will be introduced in future.

```yaml
TODO
# Mixture strategy is active for intergrated mode only.
```

### 4. Start Task Synthesis

All things configured, now we can start the task synthesis.

1. Start Enviroment Service.
2. Start Task Manager.

#### Standalone Mode

Task Manager can be started in standalone mode, which is the simplest mode for task synthesis.

To start Task Manager in standalone mode, run the following command.

```bash
TODO
```

You will see the progress of task synthesis. After the synthesis is done, it prints the path of the generated tasks.

#### Integrated Mode

In most cases, you will integrate Task Manager with AgentEvolver.

Just launch AgentEvolver to start your training.

!!! info "Independant, or integrated?"
    Task Manager can be standalone for light-weight data generation. We recommend you to tune the strategies in standalone mode first, and then use integrated mode in production, as some features are only available in AgentEvolver.

### 5. Check the Data

The synthetic tasks are stored in TODO. Check them to see if they are suitable for your agent.


## Overview of Task Manager

In data-driven model optimization, training an agent in a environment is reflected in trajectory tuning under environmental tasks. Consequently, the quality of data directly determines the agent's capability. However, acquiring and controlling the quality of training tasks in real environments is challenging.

To address this, we provide Task Manager in AgentEvolver, a general and dynamic workflow for environment exploration and task generation, along with supporting methods.

TODO 一张图

In the following sections, you will learn each component of Task Manager, and how to wisely extend them to your needs.


## Environment Profiling

Environment Profile describes concepts in the environment through entities, attributes, and operations. Similar to object-oriented programming and databases, we consider these to be the essential components of a environment.

- Entity: Represents an object in the environment.
- Attributes: Properties that describe the entity.
- Operations: Actions applicable to the entity.

For example, a simple File entity with its attributes and operations could be defined as:

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

The Profile itself is not strictly defined. Leveraged the capabilities of LLMs, it can be constructed at different levels of granularity: as a single file system entity, as individual file entities, or even as separate entities for each file type. The choice of granularity is a trade-off between human effort and the capabilities of the LLM in practice.

Based on the Environment Profile, Task Manager can recognize the concepts present in the environment, explore the information carried by entities, and leverage the available operations to process this information. These operations can be combined to form candidate solutions for real-world problems.

In addition to the Environment Profile, users may optionally provide a User Preference. A Preference describes expectations about the agent's capabilities, such as the desired task difficulty or types of tasks to be solved.

### Write a Profile

There are two ways to write a profile: using JSON (recommended) or using Python.


Top-level structure:

```json
{
  "name": string,                 // Required. User/profile name, e.g. "Alice"
  "background": string,           // Required. User background or usage scenario
  "entities": [ ... ],            // Required. List of entities in the environment (see below)
  "task_preference": {
    "num_entities": integer,         // Maximum/target number of entity types involved in a task
    "num_opts": integer,             // Maximum/target number of operations allowed in a single task or workflow
    "relation_difficulty": integer   // Complexity level of relationships/dependencies between entities
  }
}
```

`entities` is an array, each element describing an entity type (e.g. `file`, `directory`).

```json
{
  "name": string,              // Required. Entity name (lowercase recommended), e.g. "file"
  "description": string,       // Required. Human-readable description of the entity
  "attrs": {
    "name": "The description to this attr"
  },
  "opts": [
    {
      "name": string,          // Required. Operation name (verb), e.g. "create", "delete"
      "description": string    // Required. Human-readable description of the operation
    }
  ]
}
```

A minimal usable example (no comments):

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

If you prefer Python, you can find examples in the package `TODO`.


## Task Derivation


Task Derivation constitutes the initial stage of synthetic task generation. Its purpose is to transform the conceptual knowledge provided by the Environment Profile into preliminary task drafts. In this stage, the system applies exploration-synthesis strategies to traverse the environment, construct trajectories, and abstract them into structured task descriptions.

The primary objectives of derivation are:

1. Exploration – to systematically or stochastically cover the environment space.
2. Summarization – to convert exploration trajectories into concise, well-formed candidate tasks.

The following derivation strategies are available:

- RandomWalk Strategy – Performs unbiased random exploration, producing a diverse set of candidate tasks.
- More strategies will be introduced in future...

### RandomWalk Strategy

RandomWalk Strategy is a simple and effective strategy for task derivation. It leverages curiosity and exploration to generate a wide range of candidate tasks.

The parameters of RandomWalk Strategy are:

```yaml
task_manager:
  # ......
  strategy: random
  strategy_args:
    max_explore_step: 30              # Maximum number of steps to explore
    max_llm_retries: 6                # Maximum number of LLM retries
    env_url: ${env_service.env_url}   # Environment Service URL
    exploration_llm_temperature: 1.0  # LLM temperature
    exploration_llm_top_p: 1.0        # LLM top-p
    exploration_llm_top_k: 100        # LLM top-k
```


## Task Curation

Task Curation is responsible for ensuring the quality and diversity of synthetic tasks generated during derivation. It operates by applying filters to discard unsuitable tasks and mixture strategies to balance task distributions.


- Filters
    - DeduplicationFilter: Removes duplicated or highly similar tasks.
    - FeasibilityFilter: Eliminates tasks that cannot be executed in the given environment.
- Mixture Strategies
    - UnifiedMixtureStrategy: Integrates tasks from multiple sources to maintain distributional balance.

The goals of task curation are:
- Quality assurance – Valid, feasible, and logically consistent tasks.
- Diversity preservation – Avoiding over-concentration on a single class of tasks.
- Dynamic control – Adjusting task sets during training according to agent performance.

### DeduplicationFilter

DeduplicationFilter is a filter that removes duplicated or highly similar tasks. It is designed to remove tasks that are likely to be redundant or similar to each other, thus reducing the number of tasks and improving task diversity.

The filter is enabled by default.

### FeasibilityFilter




## Synthetic Reward


To facilitate direct training without requiring user-defined reward functions, Task Manager provides a built-in synthetic reward mechanism as a fallback. Characteristics of the synthetic reward:

- Generality – Designed to provide reasonable feedback across a wide range of environments.
- Zero-configuration – No additional user specification required.
- Extensibility – Can be combined with or replaced by custom reward functions.

Typical reward components include:
- Relevance check - Whether the trajectory is relevant to the task.
- Success Check with Ref. GT – Whether the task has been completed successfully.
- Efficiency check with Ref. GT - Whether the task has been completed within reasonable steps.

> While the built-in reward achieves good general performance, it is strongly recommended to implement custom reward functions tailored to your application for optimal results.

To use the built-in synthetic reward, simply set `synthetic_grader: llm` in the configuration file.

```yaml
task_manager:
  # ......
  grader:
    original_grader: env    # use environment reward for original tasks
    synthetic_grader: llm   # use synthetic reward for synthetic tasks
```


## Extend Task Manager

Task Manager is designed as a modular and extensible framework, enabling users to adapt it to a wide variety of training scenarios.

Extension points include:

- Environment Profiling – Define new entities, attributes, operations, or adjust the granularity of profiles.
- Task Derivation – Implement novel exploration or synthesis strategies.
- Task Curation – Introduce additional filters and mixture strategies (e.g., semantic similarity–based filtering).
- Reward Functions – Replace or augment the synthetic reward to reflect domain-specific success criteria.


TODO waiting for refactoring