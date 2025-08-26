from .task_manager import TaskManager
from .base import TaskObjectiveRetrieval,NaiveTaskObjectiveRetrieval
import hydra

__all__= [
    "TaskManager",
    "TaskObjectiveRetrieval",
    "NaiveTaskObjectiveRetrieval"
]

@hydra.main(config_path="../../../config", config_name="beyond_agent_dataflow", version_base=None)
def run_task_manager(config):
    from beyondagent.client.llm_client import DashScopeClient
    from beyondagent.module.task_manager.data_mixture import UnifiedMixtureStrategy
    from verl.utils.fs import copy_to_local
    from verl.utils.tokenizer import hf_tokenizer
    from beyondagent.client.env_client import EnvClient
    local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get('use_shm', False))
    
    llm_client=DashScopeClient(model_name=config.task_manager.llm_client)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
    ta=TaskManager(
        config=config,
        exploration_strategy=config.task_manager.strategy,
        exploration_strategy_args=config.task_manager.strategy_args,
        llm_client=llm_client, # or use policy model
        old_retrival=NaiveTaskObjectiveRetrieval(),
        mixture_strategy=UnifiedMixtureStrategy(
            use_original=config.task_manager.mixture.use_original_tasks,
            synthetic_ratio=config.task_manager.mixture.synthetic_data_ratio,
            shuffle=config.task_manager.mixture.shuffle,
            seed=42,
            ),
        reward_config=config.task_manager.grader,
        tokenizer=tokenizer,
        env_service_url=config.env_service.env_url,
        num_explore_threads=config.task_manager.num_explore_threads,
        n=config.task_manager.n,
    )
    env_client=EnvClient(config.env_service.env_url)
    seed_tasks=ta.load_tasks_from_environment(env_client,env_type=config.env_service.env_type,split="train")
    print("#seed_tasks: ",seed_tasks)
    generated=ta.generate_task(ta._tasks,show_progress=True)
    print(len(generated))

if __name__=="__main__":
    run_task_manager()