from agentevolver.module.task_manager import run_task_manager
import hydra


@hydra.main(config_path="../../../config", config_name="script_config", version_base=None)
def main(config):
    run_task_manager(config)

if __name__ == "__main__":
    main()