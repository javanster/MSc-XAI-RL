import os
import random
from typing import cast

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.api.models import Sequential
from keras.api.saving import load_model
from tqdm import tqdm

from rl_tcav import (
    BaselineBinaryConceptProbe,
    BaselineContinuousConceptProbe,
    BinaryConcept,
    ContinuousConcept,
    ModelActivationObtainer,
)

from ...constants import MODEL_OF_INTEREST_NAME, MODEL_OF_INTEREST_PATH
from .constants import (
    CONCEPT_DATA_APPROACH_PATH_BASE,
    CONCEPT_NAMES,
    CONCEPT_PREFIXES,
    VALIDATION_DATASET_PATHS,
    VALIDATION_LABEL_SET_PATHS,
)

random.seed(28)
np.random.seed(28)
tf.random.set_seed(28)


def load_binary_concept_examples_with_prefix(approach_path, file_prefix):
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
            elif filename.startswith(file_prefix) and filename.endswith("negative_examples.npy"):
                neg_examples_file_path = os.path.join(batch_path, filename)

        if pos_examples_file_path and neg_examples_file_path:
            try:
                pos_arr = np.load(pos_examples_file_path)
                neg_arr = np.load(neg_examples_file_path)
                data.append((pos_arr, neg_arr))
            except Exception as e:
                print(f"Error loading: {e}")
        else:
            print(f"Positive and negative examples for file prefix {file_prefix} was not found...")
    return data


def load_continuous_concept_examples_with_prefix(approach_path, file_prefix):
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


def _balance_binary_example_sets(pos_examples, neg_examples):
    min_len = min(len(pos_examples), len(neg_examples))
    if min_len < 10:
        return None, None
    pos_sample_indices = np.random.choice(len(pos_examples), size=min_len, replace=False)
    neg_sample_indices = np.random.choice(len(neg_examples), size=min_len, replace=False)
    pos_examples = pos_examples[pos_sample_indices]
    neg_examples = neg_examples[neg_sample_indices]
    return pos_examples, neg_examples


def _remove_binary_val_set_duplicates(concept_data, validation_dataset):
    validation_set = set(arr.tobytes() for arr in validation_dataset)
    new_concept_arr = []

    for pos_examples, neg_examples in concept_data:
        new_pos_examples_arr = [ex for ex in pos_examples if ex.tobytes() not in validation_set]
        new_neg_examples_arr = [ex for ex in neg_examples if ex.tobytes() not in validation_set]
        new_concept_arr.append((np.array(new_pos_examples_arr), np.array(new_neg_examples_arr)))

    return new_concept_arr


def _remove_continuous_val_set_duplicates(concept_data, validation_dataset):
    validation_set = set(arr.tobytes() for arr in validation_dataset)
    new_concept_arr = []

    for examples, labels in concept_data:
        new_examples_list = []
        new_labels_list = []
        for ex, label in zip(examples, labels):
            if ex.tobytes() not in validation_set:
                new_examples_list.append(ex)
                new_labels_list.append(label)
        new_concept_arr.append((np.array(new_examples_list), np.array(new_labels_list)))

    return new_concept_arr


def _create_binary_expanding_sample_sets(pos_examples, neg_examples):
    n = len(pos_examples)
    sample_sizes = []
    size = 10

    while size < n:
        sample_sizes.append(size)
        size *= 2
    if not sample_sizes or sample_sizes[-1] != n:
        sample_sizes.append(n)

    samples = []
    indices = np.random.permutation(n)

    for s in sample_sizes:
        selected_indices = indices[:s]
        pos_sample = pos_examples[selected_indices]
        neg_sample = neg_examples[selected_indices]
        samples.append((pos_sample, neg_sample))

    return samples


def _create_continuous_expanding_sample_sets(examples, labels):
    if len(examples) != len(labels):
        raise ValueError("Example length and label length do not match")

    n = len(examples)
    sample_sizes = []
    size = 10

    while size < n:
        sample_sizes.append(size)
        size *= 2
    if not sample_sizes or sample_sizes[-1] != n:
        sample_sizes.append(n)

    samples = []
    indices = np.random.permutation(n)

    for s in sample_sizes:
        selected_indices = indices[:s]
        example_sample = examples[selected_indices]
        labels_sample = labels[selected_indices]
        samples.append((example_sample, labels_sample))

    return samples


def _ensure_save_directory_exists(directory_path: str) -> None:
    directory = os.path.dirname(directory_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def obtain_cavs_by_approach(approach: str):
    approach_path = f"{CONCEPT_DATA_APPROACH_PATH_BASE}{approach}/"

    model = load_model(MODEL_OF_INTEREST_PATH)
    model = cast(Sequential, model)

    results = []

    total_iterations = len(CONCEPT_PREFIXES) * 10 * 14 * len(model.layers)
    with tqdm(total=total_iterations, unit="cav") as pbar:
        for concept_i, prefix in enumerate(CONCEPT_PREFIXES):
            concept_name = CONCEPT_NAMES[concept_i]

            if prefix.startswith("binary"):
                concept_data = load_binary_concept_examples_with_prefix(approach_path, prefix)
                validation_dataset_path = VALIDATION_DATASET_PATHS[concept_i]
                validation_labels_set_path = VALIDATION_LABEL_SET_PATHS[concept_i]
                concept_data = _remove_binary_val_set_duplicates(
                    concept_data=concept_data, validation_dataset=np.load(validation_dataset_path)
                )

                for batch_n, (pos_examples, neg_examples) in enumerate(concept_data):
                    pos_examples, neg_examples = _balance_binary_example_sets(
                        pos_examples, neg_examples
                    )
                    if pos_examples is None and neg_examples is None:
                        print("Insufficient number of examples, continuing...")
                        continue

                    samples_of_diff_sizes = _create_binary_expanding_sample_sets(
                        pos_examples, neg_examples
                    )
                    for pos_sample, neg_sample in samples_of_diff_sizes:
                        binary_concept = BinaryConcept(
                            name=concept_name,
                            environment_name="MinecartCounter",
                            positive_examples=pos_sample,
                            negative_examples=neg_sample,
                        )

                        mao = ModelActivationObtainer(model=model, input_normalization_type="image")

                        for layer_index in range(len(model.layers)):
                            probe = BaselineBinaryConceptProbe(
                                concept=binary_concept,
                                model_activation_obtainer=mao,
                                model_layer_index=layer_index,
                            )
                            probe.train_concept_probe()
                            cav = probe.extract_cav()
                            accuracy = probe.accuracy
                            concept_probe_score = probe.concept_probe_score
                            probe.validate_probe(
                                validation_dataset=np.load(validation_dataset_path),
                                validation_labels=np.load(validation_labels_set_path),
                            )
                            accuracy_on_val_set = probe.accuracy_on_validation_set
                            concept_probe_score_on_val_set = (
                                probe.concept_probe_score_on_validation_set
                            )

                            cav_dir = f"rl_tcav_data/cavs/baseline_concept_probes_experiment/minecart_counter/model_{MODEL_OF_INTEREST_NAME}/{approach}/concept_{concept_name}/batch_{batch_n}/"
                            _ensure_save_directory_exists(cav_dir)
                            cav_filename = f"{cav_dir}cav_samplesize{len(pos_sample) + len(neg_sample)}_layer{layer_index}.npy"
                            np.save(cav_filename, cav.vector)

                            results.append(
                                {
                                    "concept_name": concept_name,
                                    "batch_n": batch_n,
                                    "sample_size": len(pos_sample) + len(neg_sample),
                                    "layer_index": layer_index,
                                    "accuracy_mse": accuracy,
                                    "concept_probe_score": concept_probe_score,
                                    "accuracy_mse_on_validation_set": accuracy_on_val_set,
                                    "concept_probe_score_on_validation_set": concept_probe_score_on_val_set,
                                    "cav_filename": cav_filename,
                                }
                            )

                            pbar.update(1)

            else:
                concept_data = load_continuous_concept_examples_with_prefix(
                    approach_path=approach_path, file_prefix=prefix
                )
                validation_dataset_path = VALIDATION_DATASET_PATHS[concept_i]
                validation_labels_set_path = VALIDATION_LABEL_SET_PATHS[concept_i]
                concept_data = _remove_continuous_val_set_duplicates(
                    concept_data=concept_data, validation_dataset=np.load(validation_dataset_path)
                )

                for batch_n, (examples, labels) in enumerate(concept_data):
                    if len(examples) < 10:
                        print("Insufficient number of examples, continuing...")
                        continue

                    samples_of_diff_sizes = _create_continuous_expanding_sample_sets(
                        examples=examples, labels=labels
                    )
                    for example_sample, label_sample in samples_of_diff_sizes:
                        continuous_concept = ContinuousConcept(
                            name=concept_name,
                            environment_name="MinecartCounter",
                            examples=example_sample,
                            labels=label_sample,
                        )

                        mao = ModelActivationObtainer(model=model, input_normalization_type="image")

                        for layer_index in range(len(model.layers)):
                            probe = BaselineContinuousConceptProbe(
                                concept=continuous_concept,
                                model_activation_obtainer=mao,
                                model_layer_index=layer_index,
                            )
                            probe.train_concept_probe()
                            cav = probe.extract_cav()
                            mse = probe.concept_probe_mse
                            concept_probe_score = probe.concept_probe_score
                            probe.validate_probe(
                                validation_dataset=np.load(validation_dataset_path),
                                validation_labels=np.load(validation_labels_set_path),
                            )
                            mse_on_val_set = probe.concept_probe_mse_on_validation_set
                            concept_probe_score_on_val_set = (
                                probe.concept_probe_score_on_validation_set
                            )

                            cav_dir = f"rl_tcav_data/cavs/baseline_concept_probes_experiment/minecart_counter/model_{MODEL_OF_INTEREST_NAME}/{approach}/concept_{concept_name}/batch_{batch_n}/"
                            _ensure_save_directory_exists(cav_dir)
                            cav_filename = f"{cav_dir}cav_samplesize{len(example_sample)}_layer{layer_index}.npy"
                            np.save(cav_filename, cav.vector)

                            results.append(
                                {
                                    "concept_name": concept_name,
                                    "batch_n": batch_n,
                                    "sample_size": len(example_sample),
                                    "layer_index": layer_index,
                                    "accuracy_mse": mse,
                                    "concept_probe_score": concept_probe_score,
                                    "accuracy_mse_on_validation_set": mse_on_val_set,
                                    "concept_probe_score_on_validation_set": concept_probe_score_on_val_set,
                                    "cav_filename": cav_filename,
                                }
                            )

                            pbar.update(1)

    df_results = pd.DataFrame(results)
    df_dir = f"rl_tcav_data/cavs/baseline_concept_probes_experiment/minecart_counter/model_{MODEL_OF_INTEREST_NAME}/{approach}/"
    _ensure_save_directory_exists(df_dir)
    df_results.to_csv(
        f"{df_dir}concept_probe_scores_{approach}.csv",
        index=False,
    )
