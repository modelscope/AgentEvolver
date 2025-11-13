# Env 服务接口文档

## 概述

本文档详细描述了Env服务的所有接口，包括接口功能、参数说明、返回值格式以及多种访问方式示例。该服务提供了环境配置查询、实例管理、步骤执行等功能，适用于需要与环境交互的应用场景。

## 基础信息

- 默认服务地址：`http://localhost:8000`
- 所有接口均使用`POST`方法
- 数据交换格式：`JSON`
- 超时时间：约150-350秒（随机）

## 接口详情

### 1. 获取环境配置

#### 功能描述
获取指定环境类型的配置信息，主要返回可用的任务ID列表。

#### 参数说明
| 参数名 | 类型 | 必选 | 描述 |
|--------|------|------|------|
| env_type | string | 是 | 环境类型，如"appworld" |
| split | string | 否 | 数据分割类型，默认为"train" |
| params | dict | 否 | 额外参数 |

#### 返回值
- 类型：`List[str]`
- 描述：任务ID列表

#### 访问示例

##### EnvClient 方式
```python
from env_client import EnvClient

client = EnvClient(base_url="http://localhost:8000")
task_ids = client.get_env_profile(env_type="appworld", split="train")
print(f"Available tasks: {task_ids}")
```

##### curl 方式
```bash
curl -X POST http://localhost:8000/get_env_profile \
  -H "Content-Type: application/json" \
  -d '{
    "env_type": "appworld",
    "params": {"split": "train"}
  }'
```

##### HTTP 请求
```
POST /get_env_profile HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "env_type": "appworld",
  "params": {"split": "train"}
}
```

##### Python requests 方式
```python
import requests

url = "http://localhost:8000/get_env_profile"
data = {
    "env_type": "appworld",
    "params": {"split": "train"}
}

response = requests.post(url, json=data, timeout=350)
task_ids = response.json().get("data", [])
print(f"Available tasks: {task_ids}")
```

### 2. 获取工具信息

#### 功能描述
获取指定实例的工具信息。

#### 参数说明
| 参数名 | 类型 | 必选 | 描述 |
|--------|------|------|------|
| instance_id | string | 是 | 实例ID |
| messages | dict | 否 | 消息内容 |
| params | dict | 否 | 额外参数 |

#### 返回值
- 类型：`Any`
- 描述：工具信息，具体结构取决于环境实现

#### 访问示例

##### EnvClient 方式
```python
from env_client import EnvClient

client = EnvClient()
instance_id = "your-instance-id"
tools_info = client.get_tools_info(instance_id=instance_id)
print(f"Tools info: {tools_info}")
```

##### curl 方式
```bash
curl -X POST http://localhost:8000/get_info \
  -H "Content-Type: application/json" \
  -d '{
    "instance_id": "your-instance-id",
    "messages": {},
    "params": {}
  }'
```

##### HTTP 请求
```
POST /get_info HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "instance_id": "your-instance-id",
  "messages": {},
  "params": {}
}
```

##### Python requests 方式
```python
import requests

url = "http://localhost:8000/get_info"
data = {
    "instance_id": "your-instance-id",
    "messages": {},
    "params": {}
}

response = requests.post(url, json=data, timeout=350)
tools_info = response.json().get("data", None)
print(f"Tools info: {tools_info}")
```

### 3. 创建实例

#### 功能描述
基于指定的环境类型和任务ID创建一个新的实例。

#### 参数说明
| 参数名 | 类型 | 必选 | 描述 |
|--------|------|------|------|
| env_type | string | 是 | 环境类型 |
| task_id | string | 是 | 任务ID |
| instance_id | string | 否 | 实例ID，若不提供则由系统生成 |
| params | dict | 否 | 额外参数 |

#### 返回值
- 类型：`dict`
- 结构：
  ```python
  {
      "state": [{"role": str, "content": str}, ...],  # 状态信息
      "reward": float,  # 奖励值
      "is_terminated": bool,  # 是否终止
      "info": {"instance_id": str, "task_id": str}  # 实例信息
  }
  ```

#### 访问示例

##### EnvClient 方式
```python
from env_client import EnvClient

client = EnvClient()
env_type = "appworld"
task_id = "task-123"
init_response = client.create_instance(env_type, task_id)
print(f"Created instance: {init_response['info']['instance_id']}")
```

##### curl 方式
```bash
curl -X POST http://localhost:8000/create \
  -H "Content-Type: application/json" \
  -d '{
    "env_type": "appworld",
    "task_id": "task-123",
    "instance_id": "optional-instance-id",
    "params": {}
  }'
```

##### HTTP 请求
```
POST /create HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "env_type": "appworld",
  "task_id": "task-123",
  "instance_id": "optional-instance-id",
  "params": {}
}
```

##### Python requests 方式
```python
import requests

url = "http://localhost:8000/create"
data = {
    "env_type": "appworld",
    "task_id": "task-123",
    "instance_id": "optional-instance-id",
    "params": {}
}

response = requests.post(url, json=data, timeout=350)
init_response = response.json().get("data", {})
print(f"Created instance: {init_response.get('info', {}).get('instance_id')}")
```

### 4. 执行步骤

#### 功能描述
在指定实例上执行一个动作步骤。

#### 参数说明
| 参数名 | 类型 | 必选 | 描述 |
|--------|------|------|------|
| instance_id | string | 是 | 实例ID |
| action | dict | 否 | 动作内容，格式通常为`{"role": str, "content": str}` |
| params | dict | 否 | 额外参数 |

#### 返回值
- 类型：`dict`
- 结构：
  ```python
  {
      "state": [{"role": str, "content": str}, ...],  # 执行后的状态
      "reward": float,  # 奖励值
      "is_terminated": bool,  # 是否终止
      "info": {"instance_id": str, "task_id": str}  # 实例信息
  }
  ```

#### 访问示例

##### EnvClient 方式
```python
from env_client import EnvClient

client = EnvClient()
instance_id = "your-instance-id"
action = {"role": "assistant", "content": "print('hello world')"}
result = client.step(instance_id, action)
print(f"Step result: {result}")
```

##### curl 方式
```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{
    "instance_id": "your-instance-id",
    "messages": {"role": "assistant", "content": "print('\''hello world'\'')"},
    "params": {}
  }'
```

##### HTTP 请求
```
POST /step HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "instance_id": "your-instance-id",
  "messages": {"role": "assistant", "content": "print('hello world')"},
  "params": {}
}
```

##### Python requests 方式
```python
import requests

url = "http://localhost:8000/step"
data = {
    "instance_id": "your-instance-id",
    "messages": {"role": "assistant", "content": "print('hello world')"},
    "params": {}
}

response = requests.post(url, json=data, timeout=350)
result = response.json().get("data", {})
print(f"Step result: {result}")
```

### 5. 评估实例

#### 功能描述
对指定实例进行评估，返回评估分数。

#### 参数说明
| 参数名 | 类型 | 必选 | 描述 |
|--------|------|------|------|
| instance_id | string | 是 | 实例ID |
| messages | dict | 否 | 评估相关消息 |
| params | dict | 否 | 额外参数 |

#### 返回值
- 类型：`float`
- 描述：评估分数

#### 访问示例

##### EnvClient 方式
```python
from env_client import EnvClient

client = EnvClient()
instance_id = "your-instance-id"
score = client.evaluate(instance_id)
print(f"Evaluation score: {score}")
```

##### curl 方式
```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "instance_id": "your-instance-id",
    "messages": {},
    "params": {}
  }'
```

##### HTTP 请求
```
POST /evaluate HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "instance_id": "your-instance-id",
  "messages": {},
  "params": {}
}
```

##### Python requests 方式
```python
import requests

url = "http://localhost:8000/evaluate"
data = {
    "instance_id": "your-instance-id",
    "messages": {},
    "params": {}
}

response = requests.post(url, json=data, timeout=350)
score = response.json().get("data", 0.0)
print(f"Evaluation score: {score}")
```

### 6. 释放实例

#### 功能描述
释放指定的实例资源。

#### 参数说明
| 参数名 | 类型 | 必选 | 描述 |
|--------|------|------|------|
| instance_id | string | 是 | 实例ID |

#### 返回值
- 类型：`bool`
- 描述：释放成功返回`True`，否则返回`False`

#### 访问示例

##### EnvClient 方式
```python
from env_client import EnvClient

client = EnvClient()
instance_id = "your-instance-id"
success = client.release_instance(instance_id)
print(f"Instance released: {success}")
```

##### curl 方式
```bash
curl -X POST http://localhost:8000/release \
  -H "Content-Type: application/json" \
  -d '{
    "instance_id": "your-instance-id"
  }'
```

##### HTTP 请求
```
POST /release HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "instance_id": "your-instance-id"
}
```

##### Python requests 方式
```python
import requests

url = "http://localhost:8000/release"
data = {
    "instance_id": "your-instance-id"
}

response = requests.post(url, json=data, timeout=350)
success = response.json().get("success", False)
print(f"Instance released: {success}")
```

## 完整流程示例

以下是使用Env服务的完整流程示例，涵盖从获取任务列表到释放实例的所有步骤：

### EnvClient 完整流程
```python
from env_client import EnvClient

def full_workflow_demo():
    # 初始化客户端
    client = EnvClient(base_url="http://localhost:8000")
    env_type = "appworld"
    
    # 1. 获取环境配置和任务列表
    task_ids = client.get_env_profile(env_type)
    print(f"Available tasks: {task_ids}")
    
    if not task_ids:
        print("No tasks available, exiting.")
        return
    
    # 2. 创建实例
    task_id = task_ids[0]
    init_response = client.create_instance(env_type, task_id)
    print("Initial state:", init_response)
    
    instance_id = init_response["info"]["instance_id"]
    print(f"Created instance: {instance_id}")
    
    # 3. 获取工具信息
    tools_info = client.get_tools_info(instance_id)
    print(f"Tools available: {tools_info}")
    
    # 4. 执行步骤
    action = {"role": "assistant", "content": "print('Hello from workflow!')"}
    step_result = client.step(instance_id, action)
    print(f"Step result: {step_result}")
    
    # 5. 评估实例
    score = client.evaluate(instance_id)
    print(f"Final score: {score}")
    
    # 6. 释放实例
    release_success = client.release_instance(instance_id)
    print(f"Instance released successfully: {release_success}")

if __name__ == "__main__":
    full_workflow_demo()
```

## 错误处理

1. 所有接口均实现了重试机制，默认最多重试3次
2. 重试失败时会返回预设的默认值（fallback）
3. 错误信息会记录到日志文件，默认路径为`/mnt/data/eric.czq/rl_log/error.out`
4. 可以通过设置环境变量`CLIENT_LOG_PATH`自定义日志路径

## 注意事项

1. 实例创建后应在使用完毕后及时释放，避免资源泄漏
2. 接口调用可能会有较长耗时，请确保客户端超时设置足够长
3. 不同环境类型（env_type）可能支持不同的功能和参数，请参考具体环境的文档
4. 当`is_terminated`返回`True`时，实例已结束，不应再调用`step`方法