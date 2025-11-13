import json
import tempfile
from typing import Iterator, Optional, Sequence
import uuid


from agentevolver.schema.task import Task, TaskObjective
from verl.utils.dataset.rl_dataset import RLHFDataset
from torch.utils.data import IterableDataset
import pandas as pd
from omegaconf import DictConfig, ListConfig
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.processing_utils import ProcessorMixin

def convert_to_tasks(dataset:RLHFDataset,env_type:str, grader:str)->list[Task]:
    """
    Convert RLHFDataset to Task that used by TaskManager
    """
    res=[]
    for record in dataset:
        # set query to None to disable query replacement
        task = Task(
            task_id=record["extras"]["task_id"],
            env_type=env_type,
            open_query=False,
            evaluator=grader,
        )
        res.append(task)
    
    return res

def to_rl_dataset(
    tasks: Sequence[TaskObjective],
    tokenizer: PreTrainedTokenizer,
    config: DictConfig,
    processor: Optional[ProcessorMixin] = None,
) -> RLHFDataset:
    processed_records = []

    for id,task_obj in enumerate(tasks):
        task = task_obj.task

        # build reward_model
        # however, it seems that BA does not rely on this attr
        ground_truth = [task_obj.ground_truth] if task_obj.ground_truth else []

        # build record
        record = {
            "data_source": task.env_type,
            "prompt": [{"content": str(task.task_id), "role": "user"}], # `prompt` is never used. trainer will get trajectories from env. metrics code needs this to group results.
            "reward_model": {"ground_truth": ground_truth, "style": "rule"},
            "uuid": str(uuid.uuid4()),
            "extras": {
                "task_id": task.task_id,
                "open_query": task.open_query,
                "new_query": task.query,
                "evaluator": task.evaluator,
                "ground_truth": task_obj.ground_truth, # for some graders, such as LLM Judge w/ GT
            },
        }

        processed_records.append(record)

    df = pd.DataFrame(processed_records)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        df.to_parquet(f.name)

    # convert to RLHFDataset
    return RLHFDataset([f.name], tokenizer, config, processor)


class OnflyRlDataset(IterableDataset):
    def __init__(self, release_used_dataset: bool = True):
        super().__init__()
        self._do_release_used_dataset = release_used_dataset

        self._datasets: list[RLHFDataset] = []
        self._passed_datasets_cnt = 0
        self._cur_dataset = 0
        self._cur = 0

    def __len__(self):
        pass

    def __iter__(self) -> Iterator:
        """
        Returns the iterator object itself.

        Returns:
            Iterator: The iterator object (self).
        """
        return self

    def __next__(self):
        """
        Retrieves the next item from the iterator. This method manages the iteration over multiple datasets,
        advancing to the next dataset when the current one is exhausted. It also releases used datasets if
        the `_do_release_used_dataset` flag is set.

        Returns:
            Any: The next item from the current dataset.

        Raises:
            StopIteration: If there are no more items to iterate over.
        """
        if len(self._datasets) <= self._cur_dataset:
            raise StopIteration  # ⭐ Raise StopIteration if all datasets have been processed

        this_cur = self._cur - self._passed_datasets_cnt
        if this_cur >= len(self._datasets[self._cur_dataset]):
            self._passed_datasets_cnt += len(self._datasets[self._cur_dataset])
            self._cur_dataset += 1
            this_cur = 0  # ⭐ Reset the current index for the new dataset

        if len(self._datasets) <= self._cur_dataset:
            raise StopIteration  # ⭐ Raise StopIteration if all datasets have been processed

        # release used datasets
        if self._do_release_used_dataset:
            self._release_used_dataset()  # ⭐ Release used datasets if the flag is set

        self._cur += 1  # ⭐ Increment the global cursor
        return self._datasets[self._cur_dataset][this_cur]  # ⭐ Return the next item from the current dataset

    @property
    def num_rest_data(self) -> int:
        """
        Calculate the number of remaining data points in the datasets.

        Returns:
            int: The number of remaining data points.
        """
        return sum([len(d) for d in self._datasets[self._cur_dataset :]]) - (  # ⭐ Sum the lengths of remaining datasets and adjust for the current position
            self._cur - self._passed_datasets_cnt
        )

    def append_dataset(self, dataset: RLHFDataset):
        """
        Append a new RLHFDataset to the list of datasets.

        Args:
            dataset (RLHFDataset): The dataset to be appended.
        """
        self._datasets.append(dataset)  # ⭐ Append the new dataset

    def _release_used_dataset(self):
        """
        Release the used datasets and reset the internal state to start from the first dataset again.
        """
        self._datasets = self._datasets[self._cur_dataset :]  # ⭐ Remove the used datasets
        self._cur_dataset = 0  # ⭐ Reset the current dataset index
