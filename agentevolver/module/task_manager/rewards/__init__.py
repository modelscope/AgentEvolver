import functools
from typing import Tuple, Type, Dict, Callable, Any

import functools
from dataclasses import dataclass
from typing import Type, Dict, Callable, Any, Optional


@dataclass
class _RegEntry:
    cls: Type
    singleton: bool = False
    instance: Optional[Any] = None

class RewardCalculatorManager:
    """
    A singleton class for managing and instantiating different reward calculators.
    """
    _instance = None
    _registry: Dict[str, _RegEntry] = {}

    def __new__(cls, *args, **kwargs):
        """
        Implement the Singleton pattern. If the instance does not exist, create a new instance; otherwise, return the existing instance.
        """
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def reg(self, name: str) -> Callable:
        """
        Registration decorator. Associates a class with a given name.
        Can provide the calculator as a global singleton by setting singleton=True.

        Usage:
            @calculator_manager.reg("my_calc")  # Regular (new instance created each time)
            class MyCalc: ...

            @calculator_manager.reg("my_singleton_calc", singleton=True)  # Global singleton
            class MySingletonCalc: ...

        Args:
            name: The name to register under.
            singleton: Whether to provide the instance as a global singleton.

        Returns:
            Callable: The decorator function.
        """
        def decorator(calculator_cls: Type) -> Type:
            if name in self._registry:
                print(f"'{name}' has been registered, will be overwritten by '{calculator_cls.__name__}'")
            self._registry[name] = _RegEntry(cls=calculator_cls, singleton=False, instance=None)
            return calculator_cls
        return decorator

    def get_calculator(self, name: str, *args, **kwargs) -> Any:
        """
        Factory method to retrieve an instance of a registered calculator class.

        - For normal registrations (singleton=False), a new instance is returned each time.
        - For singleton registrations (singleton=True), the first call creates and caches the instance,
          and all subsequent calls return the same cached instance, ignoring any further arguments.

        Args:
            name (str): The string name used during registration.
            args: Positional arguments to pass to the calculator class constructor.
            kwargs: Keyword arguments to pass to the calculator class constructor.

        Returns:
            Any: An instance of the corresponding calculator class (either a new one or a global singleton).

        Raises:
            ValueError: If the provided name has not been registered.
        """
        entry = self._registry.get(name)
        if not entry:
            raise ValueError(f"no reward calculator named '{name}'. available calculators: {list(self._registry.keys())}")

        if entry.singleton:
            if entry.instance is None:
                # ⭐ First creation and caching of the singleton instance
                entry.instance = entry.cls(*args, **kwargs)
            return entry.instance

        # ⭐ Non-singleton: Return a new instance each time
        return entry.cls(*args, **kwargs)

grader_manager = RewardCalculatorManager()

from .judge_with_gt import LlmAsJudgeRewardCalculatorWithGT
from .reward import LlmAsJudgeRewardCalculator
from .binary_judge import LlmAsJudgeBinaryRewardCalculator
from .binary_judge_gt import LlmAsJudgeBinaryRewardCalculatorWithGT
from .avg_judge import AvgBinaryGTJudge,AvgLlmJudge
from .env_grader import EnvGrader

__all__=[
    "LlmAsJudgeRewardCalculatorWithGT",
    "LlmAsJudgeRewardCalculator",
    "LlmAsJudgeBinaryRewardCalculator",
    "LlmAsJudgeBinaryRewardCalculatorWithGT",
    "AvgBinaryGTJudge",
    "AvgLlmJudge",
    "EnvGrader",
    "grader_manager"
]