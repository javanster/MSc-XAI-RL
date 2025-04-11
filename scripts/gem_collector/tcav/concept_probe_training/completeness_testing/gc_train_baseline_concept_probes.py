import os
import random
from typing import cast

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.api.models import Sequential
from keras.api.saving import load_model
from tqdm import tqdm

from rl_tcav import BaselineBinaryConceptProbe, BinaryConcept
from utils import ModelActivationObtainer, ensure_directory_exists

from ...constants import MODEL_OF_INTEREST_PATH
from .constants import (
    CONCEPT_EXAMPLES_DIR_PATH,
    CONCEPT_NAMES,
    CONCEPT_PREFIXES,
    CONCEPT_PROBE_DIR_PATH,
    MODEL_LAYERS_OF_INTEREST,
)

random.seed(28)
np.random.seed(28)
tf.random.set_seed(28)


def load_concept_examples_with_prefix(dir_path, file_prefix):
    pos_arr = None
    neg_arr = None

    pos_examples_file_path = None
    neg_examples_file_path = None
    for filename in sorted(os.listdir(dir_path)):
        if filename.startswith(file_prefix) and filename.endswith("positive_examples.npy"):
            pos_examples_file_path = os.path.join(dir_path, filename)
        elif filename.startswith(file_prefix) and filename.endswith("negative_examples.npy"):
            neg_examples_file_path = os.path.join(dir_path, filename)

    if pos_examples_file_path and neg_examples_file_path:
        try:
            pos_arr = np.load(pos_examples_file_path)
            neg_arr = np.load(neg_examples_file_path)
        except Exception as e:
            print(f"Error loading: {e}")
    else:
        print(f"Positive and negative examples for file prefix {file_prefix} was not found...")
    return pos_arr, neg_arr


def _balance_example_sets(pos_examples, neg_examples):
    if pos_examples is None or neg_examples is None:
        return None, None
    min_len = min(len(pos_examples), len(neg_examples))
    if min_len < 10:
        return None, None
    pos_sample_indices = np.random.choice(len(pos_examples), size=min_len, replace=False)
    neg_sample_indices = np.random.choice(len(neg_examples), size=min_len, replace=False)
    pos_examples = pos_examples[pos_sample_indices]
    neg_examples = neg_examples[neg_sample_indices]
    return pos_examples, neg_examples


def gc_train_baseline_probes():
    ensure_directory_exists(CONCEPT_PROBE_DIR_PATH)

    model = load_model(MODEL_OF_INTEREST_PATH)
    model = cast(Sequential, model)

    results = []

    total_iterations = len(CONCEPT_PREFIXES) * len(MODEL_LAYERS_OF_INTEREST)
    with tqdm(total=total_iterations, unit="concept probe") as pbar:
        for concept_i, prefix in enumerate(CONCEPT_PREFIXES):
            pos_examples, neg_examples = load_concept_examples_with_prefix(
                CONCEPT_EXAMPLES_DIR_PATH, prefix
            )
            concept_name = CONCEPT_NAMES[concept_i]

            pos_examples, neg_examples = _balance_example_sets(pos_examples, neg_examples)
            if pos_examples is None or neg_examples is None:
                print(f"Insufficient number of examples for concept {concept_name}, continuing...")
                continue

            binary_concept = BinaryConcept(
                name=concept_name,
                environment_name="GemCollector",
                positive_examples=pos_examples,
                negative_examples=neg_examples,
            )

            mao = ModelActivationObtainer(model=model, input_normalization_type="image")

            for layer_index in MODEL_LAYERS_OF_INTEREST:
                probe = BaselineBinaryConceptProbe(
                    concept=binary_concept,
                    model_activation_obtainer=mao,
                    model_layer_index=layer_index,
                )
                probe.train_concept_probe()
                accuracy = probe.accuracy
                concept_probe_score = probe.concept_probe_score

                probe_classifier = probe.binary_classifier
                probe_filename = f"{CONCEPT_PROBE_DIR_PATH}{concept_name}_layer_{layer_index}_concept_probe.keras"
                probe_classifier.save(probe_filename)

                results.append(
                    {
                        "concept_name": concept_name,
                        "layer_index": layer_index,
                        "accuracy": accuracy,
                        "concept_probe_score": concept_probe_score,
                    }
                )

                pbar.update(1)

    df_results = pd.DataFrame(results)
    df_results.to_csv(
        f"{CONCEPT_PROBE_DIR_PATH}concept_probe_scores.csv",
        index=False,
    )


if __name__ == "__main__":
    gc_train_baseline_probes()
