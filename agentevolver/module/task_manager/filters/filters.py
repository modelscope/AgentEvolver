import abc
from typing import Sequence

from agentevolver.module.task_manager.filters import TaskPostFilter
from agentevolver.schema.task import TaskObjective


class NaiveTaskPostFilter(TaskPostFilter):
    def filter(self, tasks: Sequence[TaskObjective]) -> list[TaskObjective]:
        """
        Sorts and filters a list of tasks based on their confidence and removes duplicates by comparing the similarity of their query texts.

        Args:
            tasks (Sequence[TaskObjective]): A sequence of TaskObjective objects to be filtered.

        Returns:
            list[TaskObjective]: A list of unique and high-confidence TaskObjective objects.
        """
        tasks = list(tasks)
        tasks.sort(key=lambda x: x.confidence or 0, reverse=True)  # ⭐ Sort tasks by confidence in descending order

        unique_tasks = []
        seen_queries = set()

        for i, task in enumerate(tasks):
            query = task.objective
            assert query is not None
            normalized_query = (
                query.lower().strip()
            )  # FIXME: this only supports English

            is_duplicate = False
            for seen_query in seen_queries:
                if self._check_similarity(normalized_query, seen_query):
                    is_duplicate = True
                    break

            if task.ground_truth != "" and not is_duplicate:
                unique_tasks.append(task)  # ⭐ Add task to unique_tasks if it is not a duplicate and has a non-empty ground truth
                seen_queries.add(normalized_query)

        return unique_tasks

    def _check_similarity(
        self, query1: str, query2: str, threshold: float = 0.8
    ) -> bool:
        """
        Checks the similarity between two queries based on the overlap of their words.

        Args:
            query1 (str): The first query string.
            query2 (str): The second query string.
            threshold (float, optional): The similarity threshold. Defaults to 0.8.

        Returns:
            bool: True if the similarity is above the threshold, False otherwise.
        """
        words1 = set(query1.split())
        words2 = set(query2.split())

        if not words1 or not words2:
            return False

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        similarity = len(intersection) / len(union) if union else 0  # ⭐ Calculate the similarity score
        return similarity >= threshold  # ⭐ Compare the similarity with the threshold
