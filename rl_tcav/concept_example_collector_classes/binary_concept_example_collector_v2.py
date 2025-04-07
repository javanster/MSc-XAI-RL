from typing import Callable, List, Optional

import numpy as np
from gymnasium import Env
from keras.api.models import Sequential
from tqdm import tqdm

from ..concept_classes.binary_concept import BinaryConcept
from .concept_example_collector import ConceptExampleCollector


class BinaryConceptExampleCollectorV2(ConceptExampleCollector):
    """
    A concrete implementation of ConceptExampleCollector for binary concepts.

    V2!! For every step in the environment, checks all concepts for concept presence!

    This class collects examples for a list of BinaryConcept objects by interacting
    with a Gymnasium environment. It supports different collection methods, including
    model-based greedy play, epsilon-greedy play, and random policy play.

    Attributes
    ----------
    env : Env
        The Gymnasium environment used to collect observations.
    concepts : List[BinaryConcept]
        A list of BinaryConcept objects for which examples are collected.
    max_iter : int
        The maximum number of iterations for data collection.
    normalization_callback : Callable, optional
        A function used to normalize observations before passing them to the model.
    track_example_accumulation : bool
        Whether to track the accumulation of examples over time.


    Parameters
    ----------
    env : Env
        The Gymnasium environment used to collect observations.
    concepts : List[BinaryConcept]
        A list of BinaryConcept objects for which examples will be collected.
    max_iter : int
                The maximum number of iterations for data collection.
    normalization_callback : str, optional
        The key for selecting the normalization callback. If not provided, no normalization is applied.
    track_example_accumulation : bool
        Whether to track the accumulation of examples over time.

    Raises
    ------
    ValueError
        If any provided BinaryConcept object already contains positive or negative examples.
    """

    def __init__(
        self,
        env: Env,
        concepts: List[BinaryConcept],
        max_iter: int,
        normalization_callback: Optional[str] = None,
        track_example_accumulation=False,
    ) -> None:
        super().__init__(
            env=env,
            normalization_callback=normalization_callback,
            track_example_accumulation=track_example_accumulation,
        )
        self.max_iter = max_iter
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

            self.pos_examples_accumulated = {concept.name: [] for concept in self.concepts}
            self.neg_examples_accumulated = {concept.name: [] for concept in self.concepts}

            observation, _ = self.env.reset()
            terminated = False
            truncated = False

            iterations = 0

            while any(
                len(concept.positive_examples) < example_count_per_class
                or len(concept.negative_examples) < example_count_per_class
                for concept in self.concepts
            ):

                if iterations >= self.max_iter:
                    print(
                        f"\nMax iterations ({self.max_iter}) reached when collecting examples... finishing up."
                    )
                    break

                if not (terminated or truncated):
                    for concept in self.concepts:

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

                        if self.track_example_accumulation:
                            pos_examples = self.pos_examples_accumulated[concept.name]
                            pos_examples.append(len(concept.positive_examples))
                            neg_examples = self.neg_examples_accumulated[concept.name]
                            neg_examples.append(len(concept.negative_examples))

                    action = action_selection_callback(observation)
                    observation, _, terminated, truncated, _ = self.env.step(action)
                else:
                    observation, _ = self.env.reset()
                    terminated, truncated = False, False

                iterations += 1
                pbar.set_description(f"Iteration: {iterations}/{self.max_iter}")

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

    def get_example_accumulation_data(self):
        """
        Retrieve accumulated example data.

        Returns
        -------
        tuple
            A tuple containing accumulated positive and negative examples.

        Raises
        ------
        ValueError
            If accumulation data is not available.
        """
        if not self.pos_examples_accumulated or not self.neg_examples_accumulated:
            raise ValueError(
                "\nNo accumulation data found. Did you remember to initialize the example collector with 'track_example_accumulation' set to True, and have you run collection?\n"
            )
        return self.pos_examples_accumulated, self.neg_examples_accumulated

    def save_example_accumulation_data(self, directory_path: str):
        """
        Save accumulated example data for all binary concepts to disk.

        This method saves the positive and negative example accumulation data
        as `.npy` files in the specified directory.

        Parameters
        ----------
        directory_path : str
            The path to the directory where the accumulation data will be saved.

        Raises
        ------
        ValueError
            If no accumulation data is found, either because tracking was not enabled
            or collection has not been performed.
        """
        if not self.pos_examples_accumulated or not self.neg_examples_accumulated:
            raise ValueError(
                "\nNo accumulation data found. Did you remember to initialize the example collector with 'track_example_accumulation' set to True, and have you run collection?\n"
            )
        self._ensure_save_directory_exists(directory_path=directory_path)
        for concept_name in self.pos_examples_accumulated.keys():
            positive_file_path = f"{directory_path}/binary_concept_{concept_name}_positive_examples_accumulation_data_{len(self.pos_examples_accumulated[concept_name])}_iterations.npy"
            positive_array = np.array(self.pos_examples_accumulated[concept_name])
            np.save(positive_file_path, positive_array)
            print(
                f"\nAccumulation data for positive examples of concept {concept_name} successfully saved to {positive_file_path}.\n"
            )

        for concept_name in self.neg_examples_accumulated.keys():
            negative_file_path = f"{directory_path}/binary_concept_{concept_name}_negative_examples_accumulation_data_{len(self.neg_examples_accumulated[concept_name])}_iterations.npy"
            negative_array = np.array(self.neg_examples_accumulated[concept_name])
            np.save(negative_file_path, negative_array)
            print(
                f"\nAccumulation data for negative examples of concept {concept_name} successfully saved to {negative_file_path}.\n"
            )
