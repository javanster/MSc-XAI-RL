from typing import Callable, Dict, List

from gymnasium import Env

from ..concept_classes.binary_concept import BinaryConcept
from .concept_example_collector import ConceptExampleCollector


class BinaryConceptExampleCollector(ConceptExampleCollector):
    """
    A concrete implementation of ConceptExampleCollector for binary concepts.

    This class collects examples for a list of BinaryConcept objects by interacting
    with a Gymnasium environment. It supports different collection methods and allows
    for saving the collected examples to disk.

    Parameters
    ----------
    env : Env
        The Gymnasium environment used to collect observations.
    concepts : List[BinaryConcept]
        A list of BinaryConcept objects for which examples will be collected.
    """

    def __init__(self, env: Env, concepts: List[BinaryConcept]) -> None:
        self.env: Env = env
        self.concepts: List[BinaryConcept] = concepts
        self._concept_examples_collecting_methods: Dict[str, Callable[[int], None]] = {
            "env_reset": self._env_reset_collect_concept_examples
        }

    def _env_reset_collect_concept_examples(self, example_n: int) -> None:
        """
        Collect concept examples by resetting the environment.

        This method collects positive and negative examples for all binary concepts
        by resetting the environment until the desired number of examples is collected.

        Parameters
        ----------
        example_n : int
            The number of positive and negative examples to collect for each concept.
        """
        for concept in self.concepts:

            while len(concept.positive_examples) < example_n:
                observation, _ = self.env.reset()
                concept.check_positive_presence(env=self.env, observation=observation)

            while len(concept.negative_examples) < example_n:
                observation, _ = self.env.reset()
                concept.check_negative_presence(env=self.env, observation=observation)

    def collect_examples(self, example_n: int, method: str = "env_reset") -> None:
        """
        Collect concept examples using the specified method.

        This method supports different collection strategies, determined by the
        `method` parameter, to gather examples for all binary concepts.

        Parameters
        ----------
        example_n : int
            The number of examples to collect for each concept.
        method : str, optional
            The collection method to use. Default is "env_reset".

        Raises
        ------
        ValueError
            If an invalid collection method is specified.
        """
        if method not in self._concept_examples_collecting_methods:
            raise ValueError(
                f"Invalid collection method. Must be one of{[method for method in self._concept_examples_collecting_methods.keys()]}"
            )

        self._concept_examples_collecting_methods[method](example_n)

    def save_examples(self, directory_path: str) -> None:
        """
        Save collected examples for all binary concepts to disk.

        Each concept's positive and negative examples are saved in `.npy` files
        within the specified directory.

        Parameters
        ----------
        directory_path : str
            The path to the directory where examples will be saved.
        """
        for concept in self.concepts:
            concept.save_examples(directory_path=directory_path)
