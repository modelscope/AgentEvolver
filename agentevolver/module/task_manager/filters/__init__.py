import abc
from typing import Sequence

from agentevolver.schema.task import TaskObjective


class TaskPostFilter(abc.ABC):
    @abc.abstractmethod
    def filter(self, tasks: Sequence[TaskObjective]) -> list[TaskObjective]:
        """
        Abstract method to filter a sequence of TaskObjective objects.

        Args:
            tasks (Sequence[TaskObjective]): A sequence of TaskObjective objects to be filtered.

        Returns:
            list[TaskObjective]: A list of filtered TaskObjective objects.
        """
        pass


from .filters import NaiveTaskPostFilter

__all__ = [
    "NaiveTaskPostFilter"
]