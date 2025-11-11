# Configuration Guide

## Overview

**AgentEvolver** uses a hierarchical configuration system based on [Hydra](https://github.com/facebookresearch/hydra), which allows for flexible and modular configuration management. The configuration consists of multiple layers that inherit and override settings, enabling developers to customize behavior for different use cases.

## Configuration Hierarchy

The configuration implements a three-layered inheritance model:

1. **Base Layer**: `ppo_trainer.yaml` provides system defaults
2. **Framework Layer**: `agentevolver.yaml` customizes for the framework
3. **Application Layer**: Example configuration files (`basic.yaml`/`overall.yaml`) or scripts (`run_basic.sh`/`run_overall.sh`) provide final customizations

### 1. Base Configuration: `external/config_fallback/ppo_trainer.yaml`

This file serves as the foundation of the configuration system, containing default values for all core parameters needed for training derived from veRL.

For detailed documentation, refer to the [veRL](https://github.com/volcengine/verl).

### 2. AgentEvolver Configuration: `config/agentevolver.yaml`

This file extends and customizes the base configuration for AgentEvolver features.

- Overrides algorithm parameters for training algorithm
- Configures self-questioning, -navigating, and -attributing functionality
- Defines experiment projects and names

### 3. Application Configuration

- `examples/basic.yaml`: Minimal example for getting started
- `examples/overall.yaml`: Comprehensive setup for all features
- `config/script_config.yaml`: Defines interface for script. In most of the cases, you do not need to modify this file.

These files provide final customizations for specific use cases.


## Customizing Configurations

### Method 1: New YAML Files

Create a new configuration file (e.g., `examples/my_config.yaml`):

```yaml
hydra:
  searchpath:
    - file://external/config_fallback
    - file://config

defaults:
  - ppo_trainer
  - agentevolver
  - _self_

# Custom overrides
trainer:
  experiment_name: my_experiment
  total_epochs: 50

data:
  train_batch_size: 64
```

Launch with:
```bash
python launcher.py --conf examples/my_config.yaml
```

### Method 2: Command-Line Overrides

Modify or create a script similar to `examples/run_basic.sh`:

```bash
python3 -m agentevolver.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='script_config' \
    trainer.experiment_name=my_experiment \
    data.train_batch_size=64
```