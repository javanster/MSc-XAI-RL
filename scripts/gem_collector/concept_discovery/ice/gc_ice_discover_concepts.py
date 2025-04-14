import os
import random

import numpy as np
import tensorflow as tf
from keras.api.models import Sequential
from keras.api.saving import load_model

from rl_ice import ICE
from utils import ModelActivationObtainer

if __name__ == "__main__":
    SEED = 28
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)

    last_cnn_layer_idx = 1
    n_concepts_values = [c for c in range(2, 51)]
    save_dir_path = f"rl_ice_data/ncavs/gem_collector/model_of_interest_target_class_balanced_observations/layer_{last_cnn_layer_idx}/"

    model: Sequential = load_model("models/GemCollector/denim-sweep-56/1738828803_model____0.3766avg____0.6576max____0.1570min.keras")  # type: ignore

    env_observations = np.load(
        "rl_concept_discovery_data/class_datasets_model_of_interest/gem_collector/target_class_balanced_30000_shuffled_examples.npy"
    )

    mao = ModelActivationObtainer(model=model, input_normalization_type="image")

    ice = ICE(
        model_activation_obtainer=mao,
        last_cnn_layer_idx=last_cnn_layer_idx,
        top_k_examples=10,
    )

    ice.discover_concepts(
        observations=env_observations,
        n_concepts_values=n_concepts_values,
        save_dir_path=save_dir_path,
    )
