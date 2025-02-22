from typing import Callable, List, Optional

import numpy as np
from gymnasium import Env
from keras.api.models import Sequential
from tqdm import tqdm

from ..concept_classes.binary_concept import BinaryConcept
from .concept_example_collector import ConceptExampleCollector


class BinaryConceptExampleCollector(ConceptExampleCollector):
    """
    A concrete implementation of ConceptExampleCollector for binary concepts.

    This class collects examples for a list of BinaryConcept objects by interacting
    with a Gymnasium environment. It supports different collection methods, including
    model-based greedy play, epsilon-greedy play, and random policy play.

    Attributes
    ----------
    env : Env
        The Gymnasium environment used to collect observations.
    concepts : List[BinaryConcept]
        A list of BinaryConcept objects for which examples are collected.
    max_iter_per_concept : int
        The maximum number of iterations per concept during data collection.
    normalization_callback : callable or None
        A function used to normalize observations before passing them to the model.

    Parameters
    ----------
    env : Env
        The Gymnasium environment used to collect observations.
    concepts : List[BinaryConcept]
        A list of BinaryConcept objects for which examples will be collected.
    max_iter_per_concept : int
        The maximum number of iterations per concept during data collection.
    normalization_callback : str, optional
        The key for selecting the normalization callback. If not provided, no normalization is applied.

    Raises
    ------
    ValueError
        If any provided BinaryConcept object already contains positive or negative examples.
    """

    def __init__(
        self,
        env: Env,
        concepts: List[BinaryConcept],
        max_iter_per_concept: int,
        normalization_callback: Optional[str] = None,
    ) -> None:
        super().__init__(env=env, normalization_callback=normalization_callback)
        self.max_iter_per_concept = max_iter_per_concept
        self._verify_no_concept_examples(concepts=concepts)
        self.concepts: List[BinaryConcept] = concepts

    def _verify_no_concept_examples(self, concepts: List[BinaryConcept]) -> None:
        """
        Verify that no provided BinaryConcept objects contain pre-existing examples.

        Parameters
        ----------
        concepts : List[BinaryConcept]
            A list of BinaryConcept objects to check.

        Raises
        ------
        ValueError
            If any provided BinaryConcept object already contains examples.
        """
        non_empty_concepts: List[str] = []
        for concept in concepts:
            if len(concept.positive_examples) > 0 or len(concept.negative_examples) > 0:
                non_empty_concepts.append(concept.name)

        if len(non_empty_concepts) > 0:
            raise ValueError(
                f"These concept objects already contain examples: {[cn for cn in non_empty_concepts]}. "
                "Please only provide concept objects with no examples."
            )

    def _collect_examples_by_play(
        self, example_n: int, action_selection_callback: Callable[[np.ndarray], int]
    ) -> None:
        """
        Collect examples by playing in the environment using a specified action selection method.

        Parameters
        ----------
        example_n : int
            The number of examples to collect per concept.
        action_selection_callback : Callable[[np.ndarray], int]
            A function that determines the action to take based on the current observation.
        """
        with tqdm(total=len(self.concepts) * example_n, unit="example") as pbar:
            example_count_per_class = example_n // 2

            for concept in self.concepts:
                observation, _ = self.env.reset()
                terminated = False
                truncated = False

                iterations = 0

                while (
                    len(concept.positive_examples) < example_count_per_class
                    or len(concept.negative_examples) < example_count_per_class
                ):
                    if iterations >= self.max_iter_per_concept:
                        print(
                            f"Max iterations ({self.max_iter_per_concept}) reached when collecting examples for concept {concept.name}. Moving on to the next concept..."
                        )
                        break

                    positive_presence: bool = False
                    negative_presence: bool = False

                    if len(concept.positive_examples) < example_count_per_class:
                        positive_presence = concept.check_positive_presence(
                            env=self.env, observation=observation
                        )
                    if (
                        not positive_presence
                        and len(concept.negative_examples) < example_count_per_class
                    ):
                        negative_presence = concept.check_negative_presence(
                            env=self.env, observation=observation
                        )

                    if positive_presence or negative_presence:
                        pbar.update(1)

                    if terminated or truncated:
                        observation, _ = self.env.reset()
                        terminated, truncated = False, False
                    else:
                        action = action_selection_callback(observation)
                        observation, _, terminated, truncated, _ = self.env.step(action)

                    iterations += 1

            self.env.close()

            print("\n\n=================================================")
            print("Concept example collection complete. Results:\n")

            for concept in self.concepts:
                print(
                    f"{concept.name}: {len(concept.positive_examples)} positive and {len(concept.negative_examples)} negative concept examples gathered"
                )
            print("\n=================================================\n")

    def model_greedy_play_collect_examples(self, example_n: int, model: Sequential) -> None:
        """
        Collect examples using a model-based greedy action selection strategy.

        The method selects actions greedily based on the Q-values predicted by the model.

        Parameters
        ----------
        example_n : int
            The number of examples to collect.
        model : Sequential
            The model used to determine the best actions.

        Raises
        ------
        ValueError
            If the instance does not have a normalization callback.
        """
        if not self.normalization_callback:
            raise ValueError("Normalization callback is required for model-based play.")

        def action_selection_callback(observation: np.ndarray):
            observation_reshaped = self.normalization_callback(
                np.array(observation).reshape(-1, *observation.shape)
            )
            q_values = model.predict(observation_reshaped)[0]
            action = int(np.argmax(q_values))
            return action

        self._collect_examples_by_play(
            example_n=example_n, action_selection_callback=action_selection_callback
        )

    def model_epsilon_play_collect_examples(
        self, example_n: int, model: Sequential, epsilon: float
    ) -> None:
        """
        Collect examples using a model-based epsilon-greedy action selection strategy.

        The method selects actions greedily with probability (1 - epsilon) and takes
        a random action with probability epsilon. This allows for a balance between
        exploration and exploitation during example collection.

        Parameters
        ----------
        example_n : int
            The number of examples to collect.
        model : Sequential
            The model used to predict actions.
        epsilon : float
            The exploration rate (probability of choosing a random action), must be
            in the interval (0,1).

        Raises
        ------
        ValueError
            If epsilon is not within the range (0,1).
        ValueError
            If the instance does not have a normalization callback.
        """
        if not self.normalization_callback:
            raise ValueError("Normalization callback is required for model-based play.")

        if epsilon >= 1 or epsilon <= 0:
            raise ValueError("Provided epsilon must be within the interval (0,1).")

        def action_selection_callback(observation: np.ndarray):
            if np.random.random() > epsilon:
                observation_reshaped = self.normalization_callback(
                    np.array(observation).reshape(-1, *observation.shape)
                )
                q_values = model.predict(observation_reshaped)[0]
                action = int(np.argmax(q_values))
            else:
                action = int(np.random.randint(0, self.env.action_space.n))
            return action

        self._collect_examples_by_play(
            example_n=example_n, action_selection_callback=action_selection_callback
        )

    def random_policy_play_collect_examples(self, example_n: int) -> None:
        """
        Collect examples using a random action selection strategy.

        Actions are chosen randomly from the environment's action space.

        Parameters
        ----------
        example_n : int
            The number of examples to collect.
        """

        def action_selection_callback(_: np.ndarray):
            action = int(np.random.randint(0, self.env.action_space.n))
            return action

        self._collect_examples_by_play(
            example_n=example_n, action_selection_callback=action_selection_callback
        )

    def save_examples(self, directory_path: str) -> None:
        """
        Save collected examples for all binary concepts to disk.

        Each concept's positive and negative examples are saved as `.npy` files
        in the specified directory.

        Parameters
        ----------
        directory_path : str
            The path to the directory where examples will be saved.
        """
        for concept in self.concepts:
            concept.save_examples(directory_path=directory_path)
