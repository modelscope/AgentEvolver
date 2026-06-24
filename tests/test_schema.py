import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agentevolver.schema.task import Task, TaskObjective


class TestTask:
    """Unit tests for the Task schema."""

    def test_task_minimal_construction(self):
        task = Task(task_id="task-1")
        assert task.task_id == "task-1"
        assert task.env_type == "appworld"

    def test_open_query_defaults_to_false(self):
        """open_query should default to False so callers do not need to set it."""
        task = Task(task_id="task-1")
        assert task.open_query is False

    def test_open_query_can_be_overridden(self):
        task = Task(task_id="task-2", open_query=True)
        assert task.open_query is True

    def test_ground_truth_is_optional(self):
        task_with_gt = Task(
            task_id="task-3",
            ground_truth="4",
        )
        assert task_with_gt.ground_truth == "4"

        task_without_gt = Task(task_id="task-4")
        assert task_without_gt.ground_truth is None

    def test_query_defaults_to_none(self):
        task = Task(task_id="task-5")
        assert task.query is None


class TestTaskObjective:
    """Unit tests for the TaskObjective wrapper."""

    def test_task_objective_proxies_ground_truth(self):
        task = Task(
            task_id="task-6",
            ground_truth="Roses are red.",
            open_query=True,
        )
        objective = TaskObjective(task=task)

        assert objective.ground_truth == "Roses are red."
        assert objective.task.open_query is True

    def test_task_objective_exposes_query_as_objective(self):
        task = Task(
            task_id="task-7",
            query="Please summarize.",
        )
        objective = TaskObjective(task=task)
        assert objective.objective == "Please summarize."

    def test_task_objective_objective_without_query(self):
        task = Task(
            task_id="task-8",
        )
        objective = TaskObjective(task=task)
        assert objective.objective is None

    def test_task_objective_ground_truth_setter(self):
        task = Task(task_id="task-9")
        objective = TaskObjective(task=task)

        objective.ground_truth = "Updated ground truth."
        assert task.ground_truth == "Updated ground truth."
        assert objective.ground_truth == "Updated ground truth."

    def test_task_objective_dict_roundtrip(self):
        task = Task(
            task_id="task-10",
            ground_truth="Gravity is a force.",
            open_query=True,
        )
        objective = TaskObjective(task=task, confidence=0.9, reward=1.0)

        data = objective.model_dump()
        restored = TaskObjective(**data)

        assert restored.task.task_id == objective.task.task_id
        assert restored.task.ground_truth == objective.task.ground_truth
        assert restored.task.open_query == objective.task.open_query
        assert restored.confidence == 0.9
        assert restored.reward == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
