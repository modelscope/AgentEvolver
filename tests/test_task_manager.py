"""
1. 启动一个 env_service
2. 运行该文件

"""

import hydra

from agentevolver.client.llm_client import DashScopeClient
from agentevolver.module.task_manager import NaiveTaskObjectiveRetrieval, TaskManager
from agentevolver.schema.task import Task
from agentevolver.client.env_client import EnvClient


@hydra.main(
    config_path="../config",
    config_name="script_config",
    version_base=None,
)
def test_get_task_from_cache(config):
    import transformers
    import json
    from torch.utils.data import DataLoader
    from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True
    )
    manager = TaskManager(
        config,
        DashScopeClient(),
        NaiveTaskObjectiveRetrieval(),
        tokenizer=tokenizer,
        env_service_url="http://localhost:8000",
        max_explore_step=5,
        max_llm_retries=3,
        cache_dir='./task_manager_cache',
        num_explore_threads=2,
        n=1,
    )
    
    manager.load_tasks_from_environment(EnvClient("http://localhost:8000"),env_type="webshop",split='train')
    import pdb;pdb.set_trace()
    dataset=manager.debug_get_original_seed_dataset(config=config.data,tokenizer=tokenizer,processor=None)
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=False, collate_fn=default_collate_fn
    )


@hydra.main(
    config_path="../config",
    config_name="beyond_agent_dataflow",
    version_base=None,
)
def test_get_task_from_env(config):
    import transformers
    import json
    from torch.utils.data import DataLoader
    from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True
    )
    manager = TaskManager(
        config,
        DashScopeClient(),
        NaiveTaskObjectiveRetrieval(),
        tokenizer=tokenizer,
        env_service_url="http://localhost:8000",
        max_explore_step=5,
        max_llm_retries=3,
        cache_dir='./task_manager_cache',
        num_explore_threads=2,
        n=1,
    )
    
    manager.load_tasks_from_environment(EnvClient("http://localhost:8000"),env_type="appworld",split='train')
    import pdb;pdb.set_trace()
    dataset=manager.debug_get_original_seed_dataset(config=config.data,tokenizer=tokenizer,processor=None)
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=False, collate_fn=default_collate_fn
    )

    print("ready to retrieve data")
    for data in dataloader:
        import pdb

        pdb.set_trace()


@hydra.main(
    config_path="../config",
    config_name="beyond_agent_dataflow",
    version_base=None,
)
def test(config):
    import transformers
    import json
    from torch.utils.data import DataLoader
    from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True
    )
    manager = TaskManager(
        config,
        DashScopeClient(),
        NaiveTaskObjectiveRetrieval(),
        tokenizer=tokenizer,
        env_service_url="http://localhost:8000",
        max_explore_step=5,
        max_llm_retries=3,
        cache_dir='./task_manager_cache',
        num_explore_threads=2,
        n=1,
    )
    task = Task(task_id="0a9d82a_1", env_type="appworld")
    tasks = [task] * 2
    manager.load_tasks(tasks)
    dataset = manager.debug_get_original_seed_dataset(config=config.data,tokenizer=tokenizer,processor=None)
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=False, collate_fn=default_collate_fn
    )

    print("ready to retrieve data")
    for data in dataloader:
        import pdb

        pdb.set_trace()


if __name__ == "__main__":
    test_get_task_from_cache()