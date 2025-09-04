import abc

from loguru import logger

from beyondagent.module.task_manager.base import LlmClient, TaskObjectiveRetrieval
from beyondagent.module.task_manager.user_profiles import UserProfile
from beyondagent.schema.task import Task, TaskObjective
from beyondagent.schema.trajectory import Trajectory


class TaskExploreStrategy(abc.ABC):
    """The abstract class of exploration strategy used in Task Manager for task generation.
    
    It provides nescessary contexts.
    """
    def _inject_deps(self,old_retrival: TaskObjectiveRetrieval,llm_client_explore: LlmClient, llm_client_summarize: LlmClient, user_profile:UserProfile):
        self._old_retrival = old_retrival
        # TODO: where should I init the llm client
        self._llm_client_explore=llm_client_explore
        self._llm_client_summarize=llm_client_summarize
        self._user_profile=user_profile
    
    @property
    def llm_client_explore(self):
        if not hasattr(self,"_llm_client_explore"):
            raise AttributeError("llm_client is not injected")
        return self._llm_client_explore
    
    @property
    def llm_client_summarize(self):
        if not hasattr(self,"_llm_client_summarize"):
            raise AttributeError("llm_client is not injected")
        return self._llm_client_summarize
    
    @property
    def old_retrival(self) -> TaskObjectiveRetrieval:
        if not hasattr(self, "_old_retrival"):
            raise AttributeError("old_retrival is not injected")
        return self._old_retrival
    
    @property
    def user_profile(self) -> UserProfile:
        if not hasattr(self, "_user_profile"):
            raise AttributeError("user_profile is not injected")
        return self._user_profile
    
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


