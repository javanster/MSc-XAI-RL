import itertools

import gymnasium as gym
import joblib
import minecart_counter
import numpy as np
import pandas as pd
from keras.api.models import Sequential
from keras.api.saving import load_model
from tqdm import tqdm

from rl_ccm import CCM_DT
from utils import ModelActivationObtainer, ensure_directory_exists

from .constants import CCM_SCORES_DIR_PATH, MODEL_OF_INTEREST_PATH


def eval_k_means_concepts_ccm_dt(layer_i: int, k: int):
    model: Sequential = load_model(MODEL_OF_INTEREST_PATH)  # type: ignore
    env = gym.make(
        id="MinecartCounter-v2",
        scatter_minecarts=True,
    )
    mao = ModelActivationObtainer(model=model, input_normalization_type="image")

    X_train = np.load(
        "rl_ccm_data/obs_action_set/minecart_counter/kind-cosmos-35/X_train_8000_examples.npy"
    )
    X_val = np.load(
        "rl_ccm_data/obs_action_set/minecart_counter/kind-cosmos-35/X_val_2000_examples.npy"
    )

    cavs = np.load(
        f"rl_ace_data/concept_examples/k_means/minecart_counter/model_of_interest_target_class_balanced_observations/layer_{layer_i}/k_{k}_cluster_centroids.npy"
    )

    observations = np.load(
        "rl_concept_discovery_data/class_datasets_model_of_interest/minecart_counter/target_class_balanced_30000_shuffled_examples.npy"
    )
    cluster_labels = np.load(
        f"rl_ace_data/concept_examples/k_means/minecart_counter/model_of_interest_target_class_balanced_observations/layer_{layer_i}/k_{k}_cluster_labels.npy"
    )
    biases = []

    for cav_idx, cav in enumerate(cavs):
        mask_pos = cluster_labels == cav_idx
        mask_neg = cluster_labels != cav_idx

        observations_pos = observations[mask_pos]
        observations_neg = observations[mask_neg]

        if len(observations_neg) > len(observations_pos):
            rng = np.random.default_rng(seed=28 + cav_idx)
            indices = rng.choice(len(observations_neg), size=len(observations_pos), replace=False)
            observations_neg = observations_neg[indices]

        pos_activations = mao.get_layer_activations(
            layer_index=layer_i, model_inputs=observations_pos, flatten=True
        )
        neg_activations = mao.get_layer_activations(
            layer_index=layer_i, model_inputs=observations_neg, flatten=True
        )

        assert pos_activations.shape[1] == cav.shape[0], "Mismatch in activation and CAV dimensions"

        pos_scores = pos_activations @ cav
        neg_scores = neg_activations @ cav

        bias = -0.5 * (np.mean(pos_scores) + np.mean(neg_scores))
        biases.append(bias)

    biases = np.array(biases)

    results = []

    save_path = f"{CCM_SCORES_DIR_PATH}/ccm_dt/"

    ensure_directory_exists(directory_path=save_path)

    target_types = ["max_q", "all_q"]
    max_depths = [3, 4, 5]

    with tqdm(total=len(target_types) * len(max_depths), unit="score") as pbar:
        for target_type, max_depth in itertools.product(target_types, max_depths):

            Y_train = np.load(
                f"rl_ccm_data/obs_action_set/minecart_counter/kind-cosmos-35/Y_{target_type}_train_8000_examples.npy"
            )
            Y_val = np.load(
                f"rl_ccm_data/obs_action_set/minecart_counter/kind-cosmos-35/Y_{target_type}_val_2000_examples.npy"
            )

            ccm = CCM_DT(
                num_classes=env.action_space.n,
                model_activation_obtainer=mao,
                X_train=X_train,
                X_val=X_val,
                Y_train=Y_train,
                Y_val=Y_val,
                all_q=True if target_type == "all_q" else False,
                max_depth=max_depth,
            )

            completeness_score, model = ccm.train_and_eval_ccm(
                cavs=cavs,
                conv_handling="dim_reduction",
                layer_i=layer_i,
                use_sigmoid=[True for _ in range(len(cavs))],  # All concepts are binary
                biases=biases,
            )

            results.append(
                {
                    "layer": layer_i,
                    "k": k,
                    "target_type": target_type,
                    "max_depth": max_depth,
                    "completeness_score": completeness_score,
                }
            )

            joblib.dump(
                model,
                f"{save_path}best_ccm_k_means_layer_{layer_i}_k_{k}_{target_type}_max_depth_{max_depth}.joblib",
            )

            pbar.update(1)

    df = pd.DataFrame(results)
    df.to_csv(f"{save_path}k_means_concepts_completeness_scores.csv", index=False)


if __name__ == "__main__":
    eval_k_means_concepts_ccm_dt(layer_i=5, k=45)  # Based on best layer of nn ccm
