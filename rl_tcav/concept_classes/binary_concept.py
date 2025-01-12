import os
from typing import Callable, List, Optional

import numpy as np
from gymnasium import Env


class BinaryConcept:
    """
    A class to represent a binary concept with positive and negative examples.

    This class is designed to manage binary concepts, storing positive and
    negative examples of observations and allowing for their retrieval and persistence.

    Parameters
    ----------
    name : str
        The name of the binary concept.
    observation_presence_callback : Optional[Callable[[Env], bool]], optional
        A callback function that determines whether an environment observation
        is part of the positive set or not, by default None.
    positive_examples : Optional[List[np.ndarray]], optional
        A list of positive examples of observations, by default an empty list.
    negative_examples : Optional[List[np.ndarray]], optional
        A list of negative examples of observations, by default an empty list.
    """

    def __init__(
        self,
        name: str,
        observation_presence_callback: Optional[Callable[[Env], bool]] = None,
        positive_examples: Optional[List[np.ndarray]] = None,
        negative_examples: Optional[List[np.ndarray]] = None,
    ) -> None:
        self.name: str = name
        self.observation_presence_callback: Callable[[Env], bool] | None = (
            observation_presence_callback
        )
        self.positive_examples: List[np.ndarray] = (
            positive_examples if positive_examples is not None else []
        )
        self.negative_examples: List[np.ndarray] = (
            negative_examples if negative_examples is not None else []
        )

    def get_name(self) -> str:
        """
        Retrieve the name of the binary concept.

        Returns
        -------
        str
            The name of the binary concept.
        """
        return self.name

    def check_positive_presence(self, env: Env, observation: np.ndarray) -> bool:
        """
        Check if an observation belongs to the positive set and append it if so.

        Parameters
        ----------
        env : Env
            The environment where the observation occurs.
        observation : np.ndarray
            The observation to check.

        Returns
        -------
        bool
            True if the observation is added to the positive examples, False otherwise.

        Raises
        ------
        ValueError
            If `observation_presence_callback` is not provided during initialization.
        """
        if not self.observation_presence_callback:
            raise ValueError("No observation callback provided in constructor")
        if self.observation_presence_callback(env):
            self.positive_examples.append(observation)
            return True
        return False

    def check_negative_presence(self, env: Env, observation: np.ndarray) -> bool:
        """
        Check if an observation belongs to the negative set and append it if so.

        Parameters
        ----------
        env : Env
            The environment where the observation occurs.
        observation : np.ndarray
            The observation to check.

        Returns
        -------
        bool
            True if the observation is added to the negative examples, False otherwise.

        Raises
        ------
        ValueError
            If `observation_presence_callback` is not provided during initialization.
        """
        if not self.observation_presence_callback:
            raise ValueError("No observation callback provided in constructor")
        if not self.observation_presence_callback(env):
            self.negative_examples.append(observation)
            return True
        return False

    def get_positive_examples(self) -> List[np.ndarray]:
        """
        Retrieve all positive examples.

        Returns
        -------
        List[np.ndarray]
            A list of all positive examples.
        """
        return self.positive_examples

    def get_negative_examples(self) -> List[np.ndarray]:
        """
        Retrieve all negative examples.

        Returns
        -------
        List[np.ndarray]
            A list of all negative examples.
        """
        return self.negative_examples

    def get_positive_examples_len(self) -> int:
        """
        Get the number of positive examples.

        Returns
        -------
        int
            The number of positive examples.
        """
        return len(self.positive_examples)

    def get_negative_examples_len(self) -> int:
        """
        Get the number of negative examples.

        Returns
        -------
        int
            The number of negative examples.
        """
        return len(self.negative_examples)

    def save_examples(self, directory_path: str) -> None:
        """
        Save positive and negative examples to disk as `.npy` files.

        Parameters
        ----------
        directory_path : str
            The path to the directory where the examples will be saved.
            Files will be named `<name>_positive_examples.npy` and `<name>_negative_examples.npy`.

        Raises
        ------
        OSError
            If the directory cannot be created or accessed.
        """
        directory = os.path.dirname(directory_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        positive_file_path = f"{directory_path}/{self.name}_positive_examples.npy"
        negative_file_path = f"{directory_path}/{self.name}_negative_examples.npy"

        positive_array = np.array(self.positive_examples)
        negative_array = np.array(self.negative_examples)

        np.save(positive_file_path, positive_array)
        np.save(negative_file_path, negative_array)
