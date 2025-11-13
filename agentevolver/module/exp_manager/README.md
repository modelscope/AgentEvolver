## 1. 概述

**混合经验训练机制（Hybrid Experience Training）**：通过融合经验池中的历史轨迹与当前策略的在线采样，在策略更新中引入历史先验知识，减少对纯探索式rollout的依赖。

## 2. 参数说明

```yaml
hybrid_experience_training:
  enable: False
  val_rollout_expmode: "mixed"    # dev集上rollout是否加经验: ["mixed", "all", "woexp"]
  train_rollout_expmode: "mixed"  # train集上rollout是否加经验: ["mixed", "all", "woexp"]
  rollout_expratio: 0.5           # rollout时在group内部加经验的比例, train&val 保持一致
  train_sample_expmode: "keep"    # train集上rollout之后转换成训练样本保留/剔除/混合经验: ["keep", "discard", "hybrid"]
  train_sample_keepratio: 0.5     # task-level多少比例选择保留经验
  experience_template: "\n\nSome Related Experience to help you to complete the task:<EXP>{}</EXP>"  # 插入经验的模版
  
experience_maker:
  ...
  updated_freq: 0   # 更新经验池的频率(这里表示k个steps)
  val_summarizer_save: False
  
actor_rollout_ref:
  actor:
    ...
    off_cliprange_high: 1.0   # off-policy样本的cliph
```

## 3. 方法配置

|          | `val_rollout_expmode` | `train_rollout_expmode` | `train_sample_expmode` | `rollout_expratio` | `train_sample_keepratio` |
| -------- | --------------------- | ----------------------- | ---------------------- | ------------------ | ------------------------ |
| baseline | "woexp"               | "woexp"                 | -                      | 0.0             | -                        |
| EC       | "mixed"               | "mixed"                 | "keep"                 | (0, 1)             | 1.0                        |
| EI       | "mixed"               | "mixed"                 | "discard"              | (0, 1)             | 0.0                        |
| HET      | "mixed"               | "mixed"                 | "hybrid"               | (0, 1)             | (0, 1)                   |

## 4. 启动方法

```bash
# 在脚本中替换成自己的key
export SWANLAB_API_KEY="xxx"
export DASHSCOPE_API_KEY="sk-xxx"
```

### 4.1 不加经验的baseline

```bash
bash anni_scripts/run_baseline_0825.sh
```

### 4.2 加经验的baseline

- 首先需要冷启动ExperienceMaker服务，配置文件信息详见https://code.alibaba-inc.com/OpenRepo/ExperienceMaker

- 冷启动经验池可参考相关仓库。
- `EC` / `EI` / `HET`相关参数配置可见`3. 方法配置`。

```bash
bash anni_scripts/run_debug_0825.sh
```
