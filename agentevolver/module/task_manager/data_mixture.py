from abc import ABC, abstractmethod
import copy
from typing import Sequence, List, Optional
import random
import math
from agentevolver.schema.task import Task, TaskObjective
from loguru import logger


class MixtureStrategy(ABC):
    """Data Mixture Strategy Interface"""
    
    @property
    def need_synthetic(self) -> bool:
        """Whether the strategy needs synthetic data."""
        return False

    @abstractmethod
    def mix_data(self,
                 synthetic_objectives: List[TaskObjective],
                 original_tasks: Sequence[TaskObjective]) -> List[TaskObjective]:
        """
        Mixes synthetic data and original data.

        Args:
            synthetic_objectives: A list of synthetic task objectives.
            original_tasks: A sequence of original tasks.

        Returns:
            A list of mixed task objectives.
        """
        pass


class UnifiedMixtureStrategy(MixtureStrategy):
    """
    A general-purpose mixture strategy that can cover all common mixing scenarios.

    Examples:
        # Use original data exclusively
        UnifiedMixtureStrategy(use_original=True, synthetic_ratio=0)

        # Use synthetic data exclusively
        UnifiedMixtureStrategy(use_original=False, synthetic_ratio=1.0)

        # Use all original data + 0.5 times the quantity of synthetic data
        UnifiedMixtureStrategy(use_original=True, synthetic_ratio=0.5)

        # Do not use original data; use 2 times the quantity of synthetic data (relative to the original data's quantity)
        UnifiedMixtureStrategy(use_original=False, synthetic_ratio=2.0)

        # Use all original data + 1.5 times the quantity of synthetic data
        UnifiedMixtureStrategy(use_original=True, synthetic_ratio=1.5)
    """

    def __init__(self, use_original: bool = True, synthetic_ratio: float = 0.0, shuffle: bool = True, seed: Optional[int] = None):
        """
        Args:
            use_original: Whether to use the original data.
                - True: Use all original data.
                - False: Do not use original data.
            synthetic_ratio: The ratio of synthetic data, relative to the quantity of the original data.
                - 0: Do not use synthetic data.
                - 0-1: Use 'quantity of original data * synthetic_ratio' synthetic data points.
            shuffle: Whether to shuffle the data after mixing.
            seed: The random seed, used to control the randomness of sampling and shuffling.
        """
        self._use_original = use_original
        self._synthetic_ratio = synthetic_ratio
        self._shuffle = shuffle
        self._seed = seed

        if synthetic_ratio < 0:
            raise ValueError("synthetic_ratio must be non-negative")
    
    @property
    def need_synthetic(self) -> bool:
        """Whether the strategy needs synthetic data."""
        return self._synthetic_ratio > 0

    def mix_data(self,
                 synthetic_objectives: List[TaskObjective],
                 original_tasks: Sequence[TaskObjective]) -> List[TaskObjective]:
        """
        Mixes synthetic and original task objectives according to the specified strategy.

        Args:
            synthetic_objectives (List[TaskObjective]): List of synthetic task objectives.
            original_tasks (Sequence[TaskObjective]): Sequence of original task objectives.

        Returns:
            List[TaskObjective]: List of mixed task objectives.
        """
        # if seed is set, create a separate random state
        rng = random.Random(self._seed) if self._seed is not None else random

        mixed_objectives = []

        if self._use_original:
            mixed_objectives.extend(copy.deepcopy(original_tasks))
            logger.info(f"added {len(mixed_objectives)} original tasks")
        cnt_original_count = len(mixed_objectives)

        if self._synthetic_ratio > 0:
            target_synthetic_count = int(len(original_tasks) * self._synthetic_ratio)

            if target_synthetic_count > 0:
                if target_synthetic_count > len(synthetic_objectives):
                    selected_synthetic = synthetic_objectives[:]
                    logger.warning(f"not enough synthetic data: need {target_synthetic_count}, have {len(synthetic_objectives)}, using all available")
                else:
                    selected_synthetic = rng.sample(synthetic_objectives, target_synthetic_count)

                mixed_objectives.extend(selected_synthetic)
                logger.info(f"added {len(selected_synthetic)} synthetic tasks (ratio={self._synthetic_ratio})")

        if self._shuffle:
            logger.debug("shuffling data")
            rng.shuffle(mixed_objectives)

        synthetic_count = len(mixed_objectives) - cnt_original_count
        logger.info(f"final mixture: {cnt_original_count} original + {synthetic_count} synthetic = {len(mixed_objectives)} total")

        return mixed_objectives

    def __repr__(self):
        """
        Returns a string representation of the UnifiedMixtureStrategy instance.

        Returns:
            str: A string that represents the current state of the instance, including whether it uses original data, the synthetic ratio, shuffling status, and seed.
        """
        return f"UnifiedMixtureStrategy(use_original={self._use_original}, synthetic_ratio={self._synthetic_ratio}, shuffle={self._shuffle}, seed={self._seed})"  # ⭐ Generates the string representation

class OriginalOnlyStrategy(UnifiedMixtureStrategy):
    """A mixture strategy that only uses original data."""
    def __init__(self, shuffle: bool = True, seed: Optional[int] = None):
        super().__init__(use_original=True, synthetic_ratio=0.0, shuffle=shuffle, seed=seed)