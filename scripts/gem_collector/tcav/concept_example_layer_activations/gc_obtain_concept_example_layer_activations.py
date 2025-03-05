import os
import random
import re
from typing import Dict, cast

import numpy as np
from keras.api.models import Sequential
from keras.api.saving import load_model
from tqdm import tqdm

from rl_tcav import ModelActivationObtainer

from ..constants import MODEL_OF_INTEREST_NAME, MODEL_OF_INTEREST_PATH
from .constants import ACTIVATIONS_N, CONCEPT_EXAMPLE_PATHS


def load_numpy_arrays_from_paths(paths, array_filename_prefix, array_filename_ending):
    results = {}
    regex_pattern = re.compile(rf"^{array_filename_prefix}_(\d+)_{array_filename_ending}\.npy$")

    for name, path in paths.items():
        arrays = []
        if not os.path.exists(path):
            print(f"Warning: Path does not exist: {path}")
            continue

        for subdir in sorted(os.listdir(path)):
            subdir_path = os.path.join(path, subdir)
            if os.path.isdir(subdir_path):
                matching_files = [f for f in os.listdir(subdir_path) if regex_pattern.match(f)]
                for filename in matching_files:
                    array_path = os.path.join(subdir_path, filename)
                    if os.path.isfile(array_path):
                        try:
                            arr = np.load(array_path)
                            arrays.append(arr)
                        except Exception as e:
                            print(f"Error loading {array_path}: {e}")

        results[name] = arrays

    return results


def _ensure_save_directory_exists(directory_path: str) -> None:
    directory = os.path.dirname(directory_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_and_save_concept_example_activations(
    model: Sequential, approach_datasets: Dict[str, np.ndarray], concept_name: str
):
    model_activation_obtainer = ModelActivationObtainer(
        model=model, input_normalization_type="image"
    )
    with tqdm(
        total=len(model.layers) * len(approach_datasets.keys()) * 3, unit="activation_set"
    ) as pbar:

        for layer_i in range(len(model.layers)):
            for approach in approach_datasets.keys():
                observation_batches = approach_datasets[approach]
                for batch_i, observation_batch in enumerate(observation_batches):
                    sample_size = min(ACTIVATIONS_N, len(observation_batch))
                    sample_indices = np.random.choice(
                        len(observation_batch), size=sample_size, replace=False
                    )
                    random_batch_sample = observation_batch[sample_indices]
                    activations = model_activation_obtainer.get_layer_activations(
                        layer_index=layer_i, model_inputs=random_batch_sample, flatten=True
                    )
                    dir_path = f"/Volumes/MemoryBrick/MSc/rl_tcav_data/concept_example_layer_activations/gem_collector/model_{MODEL_OF_INTEREST_NAME}/{approach}/concept_{concept_name}/layer_{layer_i}/"
                    _ensure_save_directory_exists(dir_path)

                    np.save(
                        file=f"{dir_path}batch_{batch_i}_{len(activations)}_activations.npy",
                        arr=activations,
                    )
                    pbar.update(1)


if __name__ == "__main__":
    random.seed(28)
    np.random.seed(28)

    model = load_model(MODEL_OF_INTEREST_PATH)
    model = cast(Sequential, model)

    pos_aq_left_datasets = load_numpy_arrays_from_paths(
        paths=CONCEPT_EXAMPLE_PATHS,
        array_filename_prefix="binary_concept_aquamarine_left",
        array_filename_ending="positive_examples",
    )

    neg_aq_left_datasets = load_numpy_arrays_from_paths(
        paths=CONCEPT_EXAMPLE_PATHS,
        array_filename_prefix="binary_concept_aquamarine_left",
        array_filename_ending="negative_examples",
    )

    pos_lav_1_above_datasets = load_numpy_arrays_from_paths(
        paths=CONCEPT_EXAMPLE_PATHS,
        array_filename_prefix="binary_concept_lava_1_above",
        array_filename_ending="positive_examples",
    )

    neg_lav_1_above_datasets = load_numpy_arrays_from_paths(
        paths=CONCEPT_EXAMPLE_PATHS,
        array_filename_prefix="binary_concept_lava_1_above",
        array_filename_ending="negative_examples",
    )

    get_and_save_concept_example_activations(
        approach_datasets=pos_aq_left_datasets, model=model, concept_name="aquamarine_left_positive"
    )
    get_and_save_concept_example_activations(
        approach_datasets=neg_aq_left_datasets, model=model, concept_name="aquamarine_left_negative"
    )
    get_and_save_concept_example_activations(
        approach_datasets=pos_lav_1_above_datasets,
        model=model,
        concept_name="lava_1_above_positive",
    )
    get_and_save_concept_example_activations(
        approach_datasets=neg_lav_1_above_datasets,
        model=model,
        concept_name="lava_1_above_negative",
    )
