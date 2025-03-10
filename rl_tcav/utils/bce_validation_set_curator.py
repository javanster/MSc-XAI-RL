import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

random.seed(28)
np.random.seed(28)
tf.random.set_seed(28)


class BCEValidationSetCurator:
    """
    Curates a balanced validation set of Binary Concept Examples from multiple
    approaches and batches, ensuring a diverse representation of positive and negative examples.

    Parameters
    ----------
    concept_data_approach_path_base : str
        The base path to the directory containing concept data for different approaches.
    concept_file_prefix : str
        The prefix of the filenames that store positive and negative examples.
    concept_name : str
        The name of the concept being curated.
    environment_name : str
        The name of the environment the concept is associated with.
    target_size : int, optional
        The target number of examples in the validation set (default is 10,000).
    """

    APPROACHES = [
        "random_policy_play",
        "model_of_interest_greedy_play",
        "model_of_interest_epsilon0_005_play",
        "more_capable_model_greedy_play",
        "more_capable_model_epsilon0_005_play",
    ]

    def __init__(
        self,
        concept_data_approach_path_base: str,
        concept_file_prefix: str,
        concept_name: str,
        environment_name: str,
        target_size: int = 10_000,
    ) -> None:
        self.target_size = target_size
        self.concept_name = concept_name
        self.environment_name = environment_name
        self.approach_batch_sets = {}
        for approach in self.APPROACHES:
            approach_path = f"{concept_data_approach_path_base}/{approach}/"
            example_batches = self._load_concept_examples_with_prefix(
                approach_path=approach_path, file_prefix=concept_file_prefix
            )
            self.approach_batch_sets[approach] = example_batches

    def _load_concept_examples_with_prefix(self, approach_path, file_prefix):
        """
        Loads positive and negative concept examples from the specified approach path.

        Parameters
        ----------
        approach_path : str
            The directory path where batches of concept examples are stored.
        file_prefix : str
            The prefix of the filenames to identify the correct concept example files.

        Returns
        -------
        list of tuple
            A list of tuples, where each tuple contains numpy arrays of positive and negative examples.
        """
        data = []
        for batch_folder in sorted(os.listdir(approach_path)):
            batch_path = os.path.join(approach_path, batch_folder)
            if not os.path.isdir(batch_path):
                continue

            pos_examples_file_path = None
            neg_examples_file_path = None
            for filename in sorted(os.listdir(batch_path)):
                if filename.startswith(file_prefix) and filename.endswith("positive_examples.npy"):
                    pos_examples_file_path = os.path.join(batch_path, filename)
                elif filename.startswith(file_prefix) and filename.endswith(
                    "negative_examples.npy"
                ):
                    neg_examples_file_path = os.path.join(batch_path, filename)

            if pos_examples_file_path and neg_examples_file_path:
                try:
                    pos_arr = np.load(pos_examples_file_path)
                    neg_arr = np.load(neg_examples_file_path)
                    data.append((pos_arr, neg_arr))
                except Exception as e:
                    print(f"Error loading: {e}")
            else:
                print(
                    f"Positive and negative examples for file prefix {file_prefix} was not found..."
                )
        return data

    def _create_balanced_validation_set_1(self):
        """
        Creates a balanced validation set by ensuring equal distribution across approaches,
        batches, and positive/negative examples.

        Returns
        -------
        tuple
            A tuple containing:
            - validation_set : list
                A list of curated examples.
            - labels : list
                A list of corresponding labels (1 for positive, 0 for negative).
            - approach_example_counts : dict
                A dictionary tracking the count of positive and negative examples per approach and batch.
        """
        validation_set = []
        labels = []
        approach_example_counts = {}

        pos_queues = {approach: [] for approach in self.approach_batch_sets.keys()}
        neg_queues = {approach: [] for approach in self.approach_batch_sets.keys()}

        for approach in self.approach_batch_sets.keys():
            approach_example_counts[approach] = {}
            batch_set = self.approach_batch_sets[approach]
            for pos_examples, neg_examples in batch_set:
                pos_queues[approach].append(list(pos_examples))
                neg_queues[approach].append(list(neg_examples))

        for approach in pos_queues.keys():
            for batch in pos_queues[approach]:
                random.shuffle(batch)
            for batch in neg_queues[approach]:
                random.shuffle(batch)

        with tqdm(total=self.target_size, unit="example") as pbar:
            while len(validation_set) < self.target_size:
                for approach in pos_queues.keys():
                    pos_batches = pos_queues[approach]
                    neg_batches = neg_queues[approach]

                    for batch_i, (pos_batch, neg_batch) in enumerate(
                        zip(pos_batches, neg_batches, strict=True)
                    ):
                        batch_example_count = approach_example_counts[approach].get(batch_i)
                        if not batch_example_count:
                            approach_example_counts[approach][batch_i] = {
                                "pos_count": 0,
                                "neg_count": 0,
                            }

                        added = False
                        while not added:
                            if (
                                len(pos_batch) < 1
                            ):  # Ensures that all approaches and batches are balanced
                                return validation_set, labels, approach_example_counts
                            pos_example = pos_batch.pop(0)
                            if not any(np.array_equal(pos_example, arr) for arr in validation_set):
                                validation_set.append(pos_example)
                                labels.append(1)
                                added = True
                                pbar.update(1)
                                approach_example_counts[approach][batch_i]["pos_count"] += 1

                        added = False
                        while not added:
                            if len(neg_batch) < 1:  # Ensures that all approaches are balanced
                                return validation_set, labels, approach_example_counts
                            neg_example = neg_batch.pop(0)
                            if not any(np.array_equal(neg_example, arr) for arr in validation_set):
                                validation_set.append(neg_example)
                                labels.append(0)
                                added = True
                                pbar.update(1)
                                approach_example_counts[approach][batch_i]["neg_count"] += 1

        return validation_set, labels, approach_example_counts

    def _create_balanced_validation_set_2(self):
        """
        Creates a validation set by balancing across approaches, batches, and
        positive/negative examples in a round-robin manner. If a particular
        (approach, batch) group runs out of data, it is removed from the rotation.

        Returns
        -------
        tuple
            (validation_set, labels, approach_example_counts)
            Where:
                validation_set : list of ndarray
                    The curated examples.
                labels : list of int
                    Corresponding labels (1 for positive, 0 for negative).
                approach_example_counts : dict
                    Tracks how many positive and negative examples were taken
                    from each (approach, batch).
        """
        validation_set = []
        labels = []

        approach_example_counts = {}
        for approach, batch_set in self.approach_batch_sets.items():
            approach_example_counts[approach] = {}
            for batch_idx in range(len(batch_set)):
                approach_example_counts[approach][batch_idx] = {"pos_count": 0, "neg_count": 0}

        groups = []
        for approach, batch_set in self.approach_batch_sets.items():
            for batch_idx, (pos_examples, neg_examples) in enumerate(batch_set):
                pos_list = list(pos_examples)
                neg_list = list(neg_examples)
                random.shuffle(pos_list)
                random.shuffle(neg_list)
                groups.append(
                    {"approach": approach, "batch_idx": batch_idx, "pos": pos_list, "neg": neg_list}
                )

        i = 0
        with tqdm(total=self.target_size, unit="example") as pbar:
            while len(validation_set) < self.target_size and groups:
                idx = i % len(groups)
                group = groups[idx]
                approach = group["approach"]
                batch_idx = group["batch_idx"]

                if group["pos"]:
                    candidate_pos = group["pos"].pop(0)
                    if not any(np.array_equal(candidate_pos, arr) for arr in validation_set):
                        validation_set.append(candidate_pos)
                        labels.append(1)
                        approach_example_counts[approach][batch_idx]["pos_count"] += 1
                        pbar.update(1)

                if len(validation_set) >= self.target_size:
                    break

                if group["neg"]:
                    candidate_neg = group["neg"].pop(0)
                    if not any(np.array_equal(candidate_neg, arr) for arr in validation_set):
                        validation_set.append(candidate_neg)
                        labels.append(0)
                        approach_example_counts[approach][batch_idx]["neg_count"] += 1
                        pbar.update(1)

                if not group["pos"] and not group["neg"]:
                    groups.pop(idx)

                else:
                    i += 1

        return validation_set, labels, approach_example_counts

    def _ensure_save_directory_exists(self, directory_path: str) -> None:
        """
        Ensures that the directory for saving validation sets exists, creating it if necessary.

        Parameters
        ----------
        directory_path : str
            The directory path to be checked and created if needed.
        """
        directory = os.path.dirname(directory_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def curate_validation_set(self):
        """
        Generates and stores a balanced validation set using a progressive approach
        that applies two balancing strategies in sequence if needed.

        1. **Method 1:** Balances across approaches, batches, and positive/negative examples.
        2. **Method 2:** Tries to balance across approaches, positive/negative examples, and
        batches in a round robin manner, but might end up only balancing accross approach and
        positive/negative, or only positive/negative.

        If the first method fails to produce enough examples, the second method is attempted.
        The final dataset, labels, and statistics on example counts are then saved to disk.
        """
        concept_validation_set, labels, approach_example_counts = (
            self._create_balanced_validation_set_1()
        )
        method = 1

        if len(concept_validation_set) < self.target_size:
            print("Validation set curation method 1 insufficient, trying method 2...")
            concept_validation_set, labels, approach_example_counts = (
                self._create_balanced_validation_set_2()
            )
            method = 2

        print(
            f"Validation set with {len(concept_validation_set)} examples successfully curated. Storing..."
        )

        save_dir = f"rl_tcav_data/cavs/baseline_concept_probes_experiment/{self.environment_name}/validation_sets/"
        self._ensure_save_directory_exists(save_dir)

        np.save(
            f"{save_dir}{self.concept_name}_probe_validation_dataset_method{method}.npy",
            concept_validation_set,
        )
        np.save(
            f"{save_dir}{self.concept_name}_probe_validation_label_set_method{method}.npy",
            labels,
        )

        df_data = []
        for approach, batches in approach_example_counts.items():
            for batch, counts in batches.items():
                pos_count = counts.get("pos_count", 0)
                neg_count = counts.get("neg_count", 0)
                total_count = pos_count + neg_count
                df_data.append([approach, batch, pos_count, neg_count, total_count])

        df = pd.DataFrame(
            df_data, columns=["Approach", "Batch", "Pos Count", "Neg Count", "Total Count"]
        )

        df.to_csv(
            f"{save_dir}{self.concept_name}_probe_example_counts_method{method}.csv", index=False
        )

        print(f"Storing complete, see results at {save_dir}")
