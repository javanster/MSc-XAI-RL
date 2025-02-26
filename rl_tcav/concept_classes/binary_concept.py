import os
from typing import Callable, List, Optional

import numpy as np
from gymnasium import Env


class BinaryConcept:
    """
    A class to represent a binary concept with positive and negative examples.

    This class manages binary concepts by storing positive and negative observations
    and provides methods for checking, retrieving, and saving these examples while
    ensuring that duplicate examples are not stored.

    Attributes
    ----------
    name : str
        The name of the binary concept.
    environment_name : str
        The name of the environment associated with this concept.
    observation_presence_callback : Callable[[Env], bool] or None
        A callback function to determine whether an observation belongs to the positive set.
    positive_examples : List[np.ndarray]
        A list of unique positive observation examples.
    negative_examples : List[np.ndarray]
        A list of unique negative observation examples.
    """

    def __init__(
        self,
        name: str,
        environment_name: str,
        observation_presence_callback: Optional[Callable[[Env], bool]] = None,
        positive_examples: Optional[List[np.ndarray]] = None,
        negative_examples: Optional[List[np.ndarray]] = None,
    ) -> None:
        self.name: str = name
        self.environment_name = environment_name
        self.observation_presence_callback = observation_presence_callback
        self.positive_examples: List[np.ndarray] = (
            positive_examples if positive_examples is not None else []
        )
        self.negative_examples: List[np.ndarray] = (
            negative_examples if negative_examples is not None else []
        )

    def check_positive_presence(self, env: Env, observation: np.ndarray) -> bool:
        """
        Check if an observation belongs to the positive set and add it if so.

        Ensures that duplicate positive examples are not stored.

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
            If `observation_presence_callback` is not provided.
        """
        if not self.observation_presence_callback:
            raise ValueError("No observation callback provided in constructor")
        if any(np.array_equal(observation, x) for x in self.positive_examples):
            return False
        if self.observation_presence_callback(env):
            self.positive_examples.append(observation)
            return True
        return False

    def check_negative_presence(self, env: Env, observation: np.ndarray) -> bool:
        """
        Check if an observation belongs to the negative set and add it if so.

        Ensures that duplicate negative examples are not stored.

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
            If `observation_presence_callback` is not provided.
        """
        if not self.observation_presence_callback:
            raise ValueError("No observation callback provided in constructor")
        if any(np.array_equal(observation, x) for x in self.negative_examples):
            return False
        if not self.observation_presence_callback(env):
            self.negative_examples.append(observation)
            return True
        return False

    def save_examples(self, directory_path: str) -> None:
        """
        Save unique positive and negative examples to disk.

        The examples are saved as `.npy` files in the specified directory.

        Parameters
        ----------
        directory_path : str
            The path to the directory where the examples will be saved.
        """
        self._save_positive_examples(directory_path=directory_path)
        self._save_negative_examples(directory_path=directory_path)

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

    def _save_positive_examples(self, directory_path: str):
        """
        Save unique positive examples to disk.

        The positive examples are saved as a `.npy` file in the specified directory.

        Parameters
        ----------
        directory_path : str
            The path to the directory where the positive examples will be saved.
        """
        if len(self.positive_examples) == 0:
            print(f"\nNo positive examples to save for concept {self.name}, returning...\n")
            return

        self._ensure_save_directory_exists(directory_path=directory_path)
        positive_file_path = f"{directory_path}/binary_concept_{self.name}_{len(self.positive_examples)}_positive_examples.npy"
        positive_array = np.array(self.positive_examples)
        np.save(positive_file_path, positive_array)
        print(
            f"\nPositive examples of concept {self.name} successfully saved to {positive_file_path}.\n"
        )

    def _save_negative_examples(self, directory_path: str):
        """
        Save unique negative examples to disk.

        The negative examples are saved as a `.npy` file in the specified directory.

        Parameters
        ----------
        directory_path : str
            The path to the directory where the negative examples will be saved.
        """
        if len(self.negative_examples) == 0:
            print(f"\nNo negative examples to save for concept {self.name}, returning...\n")
            return

        self._ensure_save_directory_exists(directory_path=directory_path)
        negative_file_path = f"{directory_path}/binary_concept_{self.name}_{len(self.negative_examples)}_negative_examples.npy"
        negative_array = np.array(self.negative_examples)
        np.save(negative_file_path, negative_array)
        print(
            f"\nNegative examples of concept {self.name} successfully saved to {negative_file_path}.\n"
        )
