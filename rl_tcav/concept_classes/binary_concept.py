import os
from typing import Callable, List, Optional

import numpy as np
from gymnasium import Env


class BinaryConcept:
    """
    A class to represent a binary concept with positive and negative examples.

    This class is designed to manage binary concepts, storing positive and
    negative examples of observations, and allowing for their retrieval and persistence.

    Parameters
    ----------
    name : str
        The name of the binary concept.
    observation_presence_callback : Optional[Callable[[Env], bool]], optional
        A callback function that determines whether an observation belongs
        to the positive set, by default None.
    positive_examples : List[np.ndarray], optional
        A list of positive examples of observations, by default an empty list.
    negative_examples : List[np.ndarray], optional
        A list of negative examples of observations, by default an empty list.
    """

    def __init__(
        self,
        name: str,
        observation_presence_callback: Optional[Callable[[Env], bool]] = None,
        positive_examples: List[np.ndarray] = [],
        negative_examples: List[np.ndarray] = [],
    ) -> None:
        self.name: str = name
        self.observation_presence_callback: Callable[[Env], bool] | None = (
            observation_presence_callback
        )
        self.positive_examples: List[np.ndarray] = positive_examples
        self.negative_examples: List[np.ndarray] = negative_examples

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
        Save positive and negative examples to disk.

        This method saves the positive and negative examples as `.npy` files
        within the specified directory. The files are named
        `<concept_name>_<num_positive/negative_examples>_positive/negative_examples.npy`.

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

        If the specified directory does not exist, it will be created.

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
        Save positive examples to disk.

        Saves the positive examples as a `.npy` file within the specified
        directory. If there are no positive examples, the method prints a message
        and returns without saving.

        Parameters
        ----------
        directory_path : str
            The path to the directory where the positive examples will be saved.
        """
        if len(self.positive_examples) == 0:
            print("No positive examples to save, returning...")
            return

        self._ensure_save_directory_exists(directory_path=directory_path)
        positive_file_path = (
            f"{directory_path}/{self.name}_{len(self.positive_examples)}_positive_examples.npy"
        )
        positive_array = np.array(self.positive_examples)
        np.save(positive_file_path, positive_array)
        print(f"Positive concept examples successfully saved to {positive_file_path}.")

    def _save_negative_examples(self, directory_path: str):
        """
        Save negative examples to disk.

        Saves the negative examples as a `.npy` file within the specified
        directory. If there are no negative examples, the method prints a message
        and returns without saving.

        Parameters
        ----------
        directory_path : str
            The path to the directory where the negative examples will be saved.
        """
        if len(self.negative_examples) == 0:
            print("No negative examples to save, returning...")
            return

        self._ensure_save_directory_exists(directory_path=directory_path)
        negative_file_path = (
            f"{directory_path}/{self.name}_{len(self.negative_examples)}_negative_examples.npy"
        )
        negative_array = np.array(self.negative_examples)
        np.save(negative_file_path, negative_array)
        print(f"Negative concept examples successfully saved to {negative_file_path}.")
