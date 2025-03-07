import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

random.seed(28)
np.random.seed(28)
tf.random.set_seed(28)


class CCEValidationSetCurator:
    """
    Curates a balanced validation set of Continuous Concept Examples (CCE) from multiple
    approaches and batches, ensuring a diverse representation of examples with continuous labels.

    Parameters
    ----------
    concept_data_approach_path_base : str
        The base path to the directory containing concept data for different approaches.
    concept_file_prefix : str
        The prefix of the filenames that store examples and labels.
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
        Loads concept examples and continuous labels from the specified approach path.

        Parameters
        ----------
        approach_path : str
            The directory path where batches of concept examples are stored.
        file_prefix : str
            The prefix of the filenames to identify the correct concept example files.

        Returns
        -------
        list of tuple
            A list of tuples, where each tuple contains numpy arrays of examples and corresponding labels.
        """

        data = []
        for batch_folder in sorted(os.listdir(approach_path)):
            batch_path = os.path.join(approach_path, batch_folder)
            if not os.path.isdir(batch_path):
                continue

            examples_file_path = None
            labels_file_path = None
            for filename in sorted(os.listdir(batch_path)):
                if filename.startswith(file_prefix) and filename.endswith("examples.npy"):
                    examples_file_path = os.path.join(batch_path, filename)
                elif filename.startswith(file_prefix) and filename.endswith("labels.npy"):
                    labels_file_path = os.path.join(batch_path, filename)

            if examples_file_path and labels_file_path:
                try:
                    examples_arr = np.load(examples_file_path)
                    labels_arr = np.load(labels_file_path)
                    data.append((examples_arr, labels_arr))
                except Exception as e:
                    print(f"Error loading: {e}")
            else:
                print(f"Examples and labels for file prefix {file_prefix} was not found...")
        return data

    def _create_balanced_validation_set_1(self):
        """
        Creates a balanced validation set by ensuring equal distribution across approaches and batches.

        Returns
        -------
        tuple
            A tuple containing:
            - validation_set : list
                A list of curated examples.
            - val_labels : list
                A list of corresponding continuous labels.
            - approach_example_counts : dict
                A dictionary tracking the count of examples per approach and batch.
        """

        validation_set = []
        val_labels = []
        approach_example_counts = {}

        example_queues = {approach: [] for approach in self.approach_batch_sets.keys()}

        for approach in self.approach_batch_sets.keys():
            approach_example_counts[approach] = {}
            batch_set = self.approach_batch_sets[approach]
            for examples, labels in batch_set:
                example_queues[approach].append(
                    [(example, label) for (example, label) in zip(examples, labels)]
                )

        for approach in example_queues.keys():
            for batch in example_queues[approach]:
                random.shuffle(batch)

        with tqdm(total=self.target_size, unit="example") as pbar:
            while len(validation_set) < self.target_size:
                for approach in example_queues.keys():
                    batches = example_queues[approach]

                    for batch_i, batch in enumerate(batches):
                        batch_example_count = approach_example_counts[approach].get(batch_i)
                        if not batch_example_count:
                            approach_example_counts[approach][batch_i] = 0

                        added = False
                        while not added:
                            if (
                                len(batch) < 1
                            ):  # Ensures that all approaches and batches are balanced
                                return validation_set, val_labels, approach_example_counts
                            (example, label) = batch.pop(0)
                            if not any(np.array_equal(example, arr) for arr in validation_set):
                                validation_set.append(example)
                                val_labels.append(label)
                                added = True
                                pbar.update(1)
                                approach_example_counts[approach][batch_i] += 1

        return validation_set, val_labels, approach_example_counts

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
        Generates and stores a balanced validation set using the primary balancing method.

        The final dataset, labels, and statistics on example counts are saved to disk.
        """

        concept_validation_set, labels, approach_example_counts = (
            self._create_balanced_validation_set_1()
        )

        print(
            f"Validation set with {len(concept_validation_set)} examples successfully curated. Storing..."
        )

        save_dir = f"rl_tcav_data/cavs/baseline_concept_probes_experiment/{self.environment_name}/validation_sets/"
        self._ensure_save_directory_exists(save_dir)

        np.save(
            f"{save_dir}{self.concept_name}_probe_validation_dataset_method1.npy",
            concept_validation_set,
        )
        np.save(
            f"{save_dir}{self.concept_name}_probe_validation_label_set_method1.npy",
            labels,
        )

        df_data = []
        for approach, batches in approach_example_counts.items():
            for batch, count in batches.items():
                batch_labels = self.approach_batch_sets[approach][batch][1]

                mean_label = np.mean(batch_labels) if len(batch_labels) > 0 else 0
                std_label = np.std(batch_labels) if len(batch_labels) > 0 else 0

                df_data.append([approach, batch, count, mean_label, std_label])

        df = pd.DataFrame(
            df_data, columns=["Approach", "Batch", "Total Count", "Mean Label", "Std Label"]
        )

        df.to_csv(f"{save_dir}{self.concept_name}_probe_example_counts_method1.csv", index=False)

        print(f"Storing complete, see results at {save_dir}")
