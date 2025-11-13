from agentevolver.module.task_manager.env_profiles import EnvProfile
from .task_manager import TaskManager, FullDataset
from .base import TaskObjectiveRetrieval, NaiveTaskObjectiveRetrieval
import hydra

__all__ = [
    "TaskManager",
    "TaskObjectiveRetrieval",
    "NaiveTaskObjectiveRetrieval"
]


def run_task_manager(config):
    """
    Initializes and runs the task manager with the provided configuration.

    Args:
        config (DictConfig): The configuration for the task manager, loaded by Hydra.

    Returns:
        None
    """
    from agentevolver.client.llm_client import DashScopeClient
    from agentevolver.module.task_manager.data_mixture import UnifiedMixtureStrategy
    from verl.utils.fs import copy_to_local
    from verl.utils.tokenizer import hf_tokenizer
    from agentevolver.client.env_client import EnvClient
    print("loading model")
    local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get('use_shm', False))  # ⭐ Copy the model to a local path
    
    llm_client = DashScopeClient(model_name=config.task_manager.llm_client)  # ⭐ Initialize the LLM client
    tokenizer = hf_tokenizer(local_path, trust_remote_code=True)  # ⭐ Initialize the tokenizer

    print("initializing task manager")
    ta = TaskManager(
        config=config,
        exploration_strategy=config.task_manager.strategy,
        env_profile=EnvProfile.load_from_json(config.task_manager.env_profile),
        exploration_strategy_args=config.task_manager.strategy_args,
        llm_client=llm_client,  # or use policy model
        old_retrival=NaiveTaskObjectiveRetrieval(),
        mixture_strategy=UnifiedMixtureStrategy(
            use_original=False,
            synthetic_ratio=1.0,  # force synthetic_ratio to 1.0, as lazy generation is used
            shuffle=config.task_manager.mixture.shuffle,
            seed=42,
            ),
        reward_config=config.task_manager.grader,
        tokenizer=tokenizer,
        env_service_url=config.env_service.env_url,
        num_explore_threads=config.task_manager.num_explore_threads,
        n=config.task_manager.n,
    )  # ⭐ Initialize the TaskManager
    print("loading seed tasks")
    env_client = EnvClient(config.env_service.env_url)  # ⭐ Initialize the environment client
    seed_tasks = ta.load_tasks_from_environment(env_client, env_type=config.env_service.env_type, split="train")  # ⭐ Load seed tasks from the environment
    print("loaded, #seed_tasks: ", seed_tasks)

    print("generating synthetic tasks...")
    dataset = FullDataset(ta, ta._mixture_strategy, ta._reward_config, cache_path=config.task_manager.train_data_path, tokenizer=tokenizer, config=config, processor=None)

    print("finish generating, #tasks: ", len(dataset))

@hydra.main(config_path="../../../config", config_name="script_config", version_base=None)
def main(config):
    print("running task manager to generate synthetic tasks...")
    run_task_manager(config)

if __name__ == "__main__":
    main()