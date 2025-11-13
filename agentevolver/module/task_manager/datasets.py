from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from typing import (
    Iterable,
    Sequence,
)

from loguru import logger
from torch.utils.data import IterableDataset,Dataset
from agentevolver.module.task_manager.adapter import OnflyRlDataset, to_rl_dataset
from agentevolver.module.task_manager.data_mixture import MixtureStrategy, OriginalOnlyStrategy

from agentevolver.module.task_manager.task_manager import RewardProps, TaskManager
from agentevolver.schema.task import Task, TaskObjective
