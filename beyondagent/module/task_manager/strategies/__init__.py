import abc

from loguru import logger

from beyondagent.module.task_manager.base import LlmClient, TaskObjectiveRetrieval
from beyondagent.schema.task import Task, TaskObjective
from beyondagent.schema.trajectory import Trajectory


class TaskExploreStrategy(abc.ABC):
    """The abstract class of exploration strategy used in Task Manager for task generation.
    
    It provides nescessary contexts.
    """
    def _inject_deps(self,old_retrival: TaskObjectiveRetrieval,llm_client: LlmClient):
        self._old_retrival = old_retrival
        # TODO: where should I init the llm client
        self._llm_client=llm_client
    
    @property
    def llm_client(self):
        if not hasattr(self,"_llm_client"):
            raise AttributeError("llm_client is not injected")
        return self._llm_client
    
    @property
    def old_retrival(self) -> TaskObjectiveRetrieval:
        if not hasattr(self, "_old_retrival"):
            raise AttributeError("old_retrival is not injected")
        return self._old_retrival
    
    @abc.abstractmethod
    def explore(
        self, task: Task, data_id: str, rollout_id: str
    ) -> list[Trajectory]:
        """Explore the env.
        """
        pass
    
    @abc.abstractmethod
    def summarize(
        self, task: Task, trajectory: Trajectory
    ) -> list[TaskObjective]:
        pass


