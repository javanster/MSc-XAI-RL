import os
from typing import Callable, Dict, List, cast

import numpy as np
from gymnasium import Env
from tqdm import tqdm


class TcavClassLabelExampleCollector:
    """
    Collects unique observation examples for each action class from a Gymnasium environment using a callback.

    This class collects a specified number of unique examples (observations) for each output class
    (action) from a given Gymnasium environment. The examples are collected by executing actions determined
    by a callback function, and uniqueness is ensured via hashing of the observation bytes.
    Collected examples can be saved to files for later analysis.

    Parameters
    ----------
    env : Env
        The Gymnasium environment from which examples are collected.
    action_callback : Callable[[np.ndarray, int], int]
        A callback function that receives the current observation and step count and returns an action.

    Attributes
    ----------
    env : Env
        The Gymnasium environment used for collecting examples.
    action_callback : Callable[[np.ndarray, int], int]
        The callback function used to determine the action based on the current observation.
    collected_examples : Dict[int, List[np.ndarray]]
        A dictionary mapping each action (output class) to its list of collected unique observations.
    collected_examples_hashes : Dict[int, set]
        A dictionary mapping each action (output class) to a set of hashed representations of the observations,
        used to ensure uniqueness.
    """

    def __init__(
        self,
        env: Env,
        action_callback: Callable[[np.ndarray, int], int],
    ) -> None:
        self.env: Env = env
        self.action_callback = action_callback
        self.collected_examples: Dict[int, List[np.ndarray]] = {
            action: [] for action in range(env.action_space.n)
        }
        self.collected_examples_hashes: Dict[int, set] = {
            action: set() for action in range(env.action_space.n)
        }

    def _is_done(self, examples_per_output_class: int):
        """
        Check if the desired number of examples have been collected for every output class.

        Parameters
        ----------
        examples_per_output_class : int
            The required number of examples for each output class.

        Returns
        -------
        bool
            True if the number of examples for every output class is equal to or greater than
            examples_per_output_class, otherwise False.
        """
        return all(
            len(examples) >= examples_per_output_class
            for examples in self.collected_examples.values()
        )

    def collect_examples(self, examples_per_output_class: int, max_iterations: int) -> None:
        """
        Collect unique observation examples for each output class until the desired number per class
        is reached or the maximum iterations are exceeded.

        Parameters
        ----------
        examples_per_output_class : int
            The number of examples to collect for each output class.
        max_iterations : int
            The maximum number of iterations to attempt collecting examples.

        Raises
        ------
        ValueError
            If this instance has already collected examples.
        """
        if any(len(examples) > 0 for examples in self.collected_examples.values()):
            raise ValueError(
                "This instance of TcavExampleCollector has already collected examples. "
                "Please instantiate a new instance to collect examples."
            )

        with tqdm(
            total=examples_per_output_class * self.env.action_space.n, unit="example"
        ) as pbar:
            observation, _ = self.env.reset()
            observation = cast(np.ndarray, observation)
            terminated: bool = False
            truncated: bool = False
            iterations: int = 0
            step: int = 0

            while True:
                if iterations >= max_iterations:
                    print("Max iterations reached, ending collection...")
                    return

                action = self.action_callback(observation, step)
                examples_of_action = self.collected_examples[action]
                examples_hashes = self.collected_examples_hashes[action]
                obs_hash = observation.tobytes()

                if (
                    len(examples_of_action) < examples_per_output_class
                    and obs_hash not in examples_hashes
                ):
                    examples_of_action.append(observation)
                    examples_hashes.add(obs_hash)
                    pbar.update(1)

                    if self._is_done(examples_per_output_class=examples_per_output_class):
                        print("\nExample collection of class labels complete!\n")
                        return

                observation, _, terminated, truncated, _ = self.env.step(action=action)
                observation = cast(np.ndarray, observation)
                terminated = cast(bool, terminated)
                truncated = cast(bool, truncated)
                step += 1

                if terminated or truncated:
                    observation, _ = self.env.reset()
                    observation = cast(np.ndarray, observation)
                    terminated = False
                    truncated = False
                    step = 0

                iterations += 1

    def save_examples(self, directory_path: str) -> None:
        """
        Save the collected examples to files, one file per output class.

        The examples for each output class are saved as a NumPy array file. The file name indicates
        the number of examples and the output class.

        Parameters
        ----------
        directory_path : str
            The directory path where the examples will be saved.
        """
        for output_class in self.collected_examples.keys():
            self._save_examples_for_output_class(
                directory_path=directory_path, output_class=output_class
            )

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

    def _save_examples_for_output_class(self, directory_path: str, output_class: int) -> None:
        """
        Save the collected examples to files, one file per output class.

        The examples for each output class are saved as a NumPy array file. The file name indicates
        the number of examples and the output class.

        Parameters
        ----------
        directory_path : str
            The directory path where the examples will be saved.
        """
        collected_examples = self.collected_examples[output_class]

        if len(collected_examples) == 0:
            print(f"No examples of output class {output_class} to save, returning...")
            return

        self._ensure_save_directory_exists(directory_path=directory_path)
        file_path = (
            f"{directory_path}/{len(collected_examples)}_examples_output_class_{output_class}.npy"
        )
        array = np.array(collected_examples)
        np.save(file_path, array)
        print(f"Examples for output class {output_class} successfully saved to {file_path}.")
