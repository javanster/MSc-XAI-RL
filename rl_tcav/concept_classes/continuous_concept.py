import os
from typing import Callable, List, Optional

import numpy as np
from gymnasium import Env


class ContinuousConcept:
    """
    A class to represent a continuous concept with labeled examples.

    This class manages continuous concepts by storing labeled observations
    and provides methods for checking, retrieving, and saving these examples.

    Attributes
    ----------
    name : str
        The name of the continuous concept.
    environment_name : str
        The name of the environment associated with this concept.
    observation_presence_callback : Callable[[Env], float] or None
        A callback function that assigns a float label to an observation.
    examples : List[np.ndarray]
        A list of stored observations.
    labels : List[float]
        A list of corresponding labels for the stored observations.

    Parameters
    ----------
    name : str
        The name of the continuous concept.
    environment_name : str
        The name of the environment associated with this concept.
    observation_presence_callback : Callable[[Env], float], optional
        A callback function that determines the label of an observation.
    examples : List[np.ndarray], optional
        A list of initial stored observations, by default an empty list.
    labels : List[float], optional
        A list of initial labels corresponding to the stored observations,
        by default an empty list.
    """

    def __init__(
        self,
        name: str,
        environment_name: str,
        observation_presence_callback: Optional[Callable[[Env], float]] = None,
        examples: Optional[List[np.ndarray]] = None,
        labels: Optional[List[float]] = None,
    ) -> None:
        self.name: str = name
        self.environment_name = environment_name
        self.observation_presence_callback: Callable[[Env], float] | None = (
            observation_presence_callback
        )
        self.examples: List[np.ndarray] = examples if examples is not None else []
        self.labels: List[float] = labels if labels is not None else []
        self._example_hashes = (
            {self._hash_obs(obs) for obs in self.examples} if self.examples else set()
        )

    def _hash_obs(self, observation: np.ndarray) -> int:
        """
        Compute a hash for the observation based on its bytes representation.

        Parameters
        ----------
        observation : np.ndarray
            The observation to hash.

        Returns
        -------
        int
            The computed hash value.
        """
        return hash(observation.tobytes())

    def check_presence(self, env: Env, observation: np.ndarray) -> bool:
        """
        Check if an observation is already stored and add it with a label if not.

        Parameters
        ----------
        env : Env
            The environment where the observation occurs.
        observation : np.ndarray
            The observation to check.

        Returns
        -------
        bool
            True if the observation is added, False if it is already present.

        Raises
        ------
        ValueError
            If `observation_presence_callback` is not provided.
        """
        if not self.observation_presence_callback:
            raise ValueError("No observation presence callback provided in constructor")
        obs_hash = self._hash_obs(observation)
        if obs_hash in self._example_hashes:
            return False
        label = self.observation_presence_callback(env)
        self.examples.append(observation)
        self.labels.append(label)
        self._example_hashes.add(obs_hash)
        return True

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

    def save_examples(self, directory_path: str) -> None:
        """
        Save examples and their corresponding labels to disk.

        The examples are saved as `.npy` files in the specified directory.

        Parameters
        ----------
        directory_path : str
            The path to the directory where the examples will be saved.
        """
        if len(self.examples) == 0:
            print(f"\nNo examples to save for concept {self.name}, returning...\n")
            return

        self._ensure_save_directory_exists(directory_path=directory_path)
        examples_file_path = (
            f"{directory_path}/continuous_concept_{self.name}_{len(self.examples)}_examples.npy"
        )
        labels_file_path = (
            f"{directory_path}/continuous_concept_{self.name}_{len(self.examples)}_labels.npy"
        )

        examples_array = np.array(self.examples)
        labels_array = np.array(self.labels)

        np.save(examples_file_path, examples_array)
        np.save(labels_file_path, labels_array)

        print(f"\nExamples of concept {self.name} successfully saved to {examples_file_path}.\n")
        print(
            f"\nLabels of examples of concept {self.name} successfully saved to {labels_file_path}.\n"
        )
