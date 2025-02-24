import os
from abc import ABC, abstractmethod
from typing import Optional

from gymnasium import Env
from keras.api.models import Sequential

from agents import ObservationNormalizationCallbacks


class ConceptExampleCollector(ABC):
    """
    Abstract base class for collecting and managing concept examples.

    This class defines an interface for collecting concept examples from an environment
    using different strategies. It also supports applying an observation normalization
    callback if specified.

    Attributes
    ----------
    env : Env
        The environment instance used to collect examples.
    normalization_callback : callable or None
        The callback function used to normalize observations, if provided.

    Parameters
    ----------
    env : Env
        The environment instance from which to collect examples.
    normalization_callback : str, optional
        The key for selecting the normalization callback. It must be one of the keys in
        ObservationNormalizationCallbacks.normalization_callbacks. If not provided, no
        normalization is applied.
    track_example_accumulation: bool
        Whether the number of examples collected over iterations should be tracked.
        Defaults to False.

    Raises
    ------
    ValueError
        If the provided normalization_callback is not a valid key, if it is not None.
    """

    def __init__(
        self,
        env: Env,
        normalization_callback: Optional[str] = None,
        track_example_accumulation: bool = False,
    ) -> None:
        self.env: Env = env
        if (
            normalization_callback
            and normalization_callback
            not in ObservationNormalizationCallbacks.normalization_callbacks.keys()
        ):
            raise ValueError(
                f"Provided normalization_callback is not valid, please provide one of the following: "
                f"{[callback_name for callback_name in ObservationNormalizationCallbacks.normalization_callbacks.keys()]}"
            )
        if normalization_callback:
            self.normalization_callback = ObservationNormalizationCallbacks.normalization_callbacks[
                normalization_callback
            ]
        self.track_example_accumulation = track_example_accumulation

    @abstractmethod
    def model_greedy_play_collect_examples(self, example_n: int, model: Sequential) -> None:
        """
        Collect examples using a model-based greedy play strategy.

        The method should use the provided model to select actions greedily based on
        predicted outcomes.

        Parameters
        ----------
        example_n : int
            The number of examples to collect.
        model : Sequential
            The model used to determine the best actions.
        """
        pass

    @abstractmethod
    def model_epsilon_play_collect_examples(
        self, example_n: int, model: Sequential, epsilon: float
    ) -> None:
        """
        Collect examples using a model-based epsilon-greedy play strategy.

        This strategy combines greedy action selection with random exploration. The
        epsilon parameter defines the probability of taking a random action.

        Parameters
        ----------
        example_n : int
            The number of examples to collect.
        model : Sequential
            The model used to predict actions.
        epsilon : float
            The exploration rate (probability of choosing a random action).
        """
        pass

    @abstractmethod
    def random_policy_play_collect_examples(self, example_n: int) -> None:
        """
        Collect examples using a random policy play strategy.

        The method should collect examples by selecting actions randomly.

        Parameters
        ----------
        example_n : int
            The number of examples to collect.
        """
        pass

    @abstractmethod
    def save_examples(self, directory_path: str) -> None:
        """
        Save collected concept examples to disk.

        Implementations should save the examples to files in the specified directory.

        Parameters
        ----------
        directory_path : str
            The path to the directory where examples will be saved.
        """
        pass

    def _ensure_save_directory_exists(self, directory_path: str) -> None:
        """
        Ensure that the save directory exists.

        If the specified directory does not exist, it is created.

        Parameters
        ----------
        directory_path : str
            The path to the directory to check or create.
        """
        directory = os.path.dirname(directory_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
