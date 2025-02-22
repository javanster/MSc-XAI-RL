from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Union

from gymnasium import Env

CollectionMethod = Union[Callable[[int], None], Callable[[int, float], None]]


class ConceptExampleCollector(ABC):
    """
    Abstract base class for collecting and managing concept examples.

    Attributes
    ----------
    env : Env
        The environment instance used to collect examples.
    """

    def __init__(self, env: Env) -> None:
        self.env: Env = env
        self._concept_examples_collecting_methods: Dict[str, CollectionMethod] = {
            "model_greedy_play": self._model_greedy_play_collect_examples,
            "model_epsilon_play": self._model_epsilon_play_collect_examples,
            "random_policy_play": self._random_policy_play_collect_examples,
        }

    @abstractmethod
    def _model_greedy_play_collect_examples(self, example_n: int) -> None:
        """
        Collect examples using a model greedy play strategy.

        Parameters
        ----------
        example_n : int
            The number of examples to collect.
        """
        pass

    @abstractmethod
    def _model_epsilon_play_collect_examples(self, example_n: int, epsilon: float) -> None:
        """
        Collect examples using a model epsilon-greedy play strategy, where epsilon is the chance
        of taking a random action.

        Parameters
        ----------
        example_n : int
            The number of examples to collect.
        epsilon : float
            The exploration rate to use.
        """
        pass

    @abstractmethod
    def _random_policy_play_collect_examples(self, example: int) -> None:
        """
        Collect examples using a random policy play strategy.

        Parameters
        ----------
        example : int
            The number of examples to collect.
        """
        pass

    @abstractmethod
    def collect_examples(
        self, example_n: int, method: str = "model_greedy_play", epsilon: Optional[float] = None
    ) -> None:
        """
        Collect concept examples using the specified method.

        Parameters
        ----------
        example_n : int
            The number of examples to collect.
        method : str, optional
            The collection method to use (default is "model_greedy_play").
        epsilon : float, optional
            The exploration rate for epsilon play, if applicable.
        """
        pass

    @abstractmethod
    def save_examples(self, directory_path: str, example_prefix: str) -> None:
        """
        Save collected concept examples to disk.

        Parameters
        ----------
        directory_path : str
            The path to the directory where examples will be saved.
        example_prefix : str
            A prefix to use for the saved example files.
        """
        pass
