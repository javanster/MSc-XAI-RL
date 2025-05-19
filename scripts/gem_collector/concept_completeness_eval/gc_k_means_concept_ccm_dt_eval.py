import itertools

import gem_collector
import gymnasium as gym
import joblib
import numpy as np
import pandas as pd
from keras.api.models import Sequential
from keras.api.saving import load_model
from tqdm import tqdm

from rl_ccm import CCM_DT
from utils import ModelActivationObtainer, ensure_directory_exists

from .constants import CCM_SCORES_DIR_PATH, MODEL_OF_INTEREST_PATH


def eval_k_means_concepts_ccm_dt():
    model: Sequential = load_model(MODEL_OF_INTEREST_PATH)  # type: ignore
    env = gym.make(id="GemCollector-v3")
    mao = ModelActivationObtainer(model=model, input_normalization_type="image")

    results = []
    best_completeness_score = -np.inf
    best_dt_model = None
    best_params = {}

    save_path = f"{CCM_SCORES_DIR_PATH}/ccm_dt/"

    ensure_directory_exists(directory_path=save_path)

    batches = [b for b in range(100)]
    layers = [0, 1, 3, 4, 5]
    k_values = [k for k in range(2, 51)]

    with tqdm(total=len(batches) * len(layers) * len(k_values), unit="score") as pbar:
        for batch, layer_i, k in itertools.product(batches, layers, k_values):

            X_train = np.load(
                f"/Volumes/work/rl_ccm_data/obs_action_set/gem_collector/denim-sweep-56/batch_{batch}/X_train_8000_examples.npy"
            )

            X_val = np.load(
                f"/Volumes/work/rl_ccm_data/obs_action_set/gem_collector/denim-sweep-56/batch_{batch}/X_val_2000_examples.npy"
            )

            Y_train = np.load(
                f"/Volumes/work/rl_ccm_data/obs_action_set/gem_collector/denim-sweep-56/batch_{batch}/Y_all_q_train_8000_examples.npy"
            )
            Y_val = np.load(
                f"/Volumes/work/rl_ccm_data/obs_action_set/gem_collector/denim-sweep-56/batch_{batch}/Y_all_q_val_2000_examples.npy"
            )

            ccm = CCM_DT(
                num_classes=env.action_space.n,
                model_activation_obtainer=mao,
                X_train=X_train,
                X_val=X_val,
                Y_train=Y_train,
                Y_val=Y_val,
                all_q=True,
                max_depth=3,
            )

            cavs = np.load(
                f"/Volumes/work/rl_ace_data/concept_examples/k_means/gem_collector/model_of_interest_target_class_balanced_observations/batch_{batch}/layer_{layer_i}/k_{k}_cluster_centroids.npy"
            )

            observations = np.load(
                "rl_concept_discovery_data/class_datasets_model_of_interest/gem_collector/target_class_balanced_30000_shuffled_examples.npy"
            )

            cluster_labels = np.load(
                f"/Volumes/work/rl_ace_data/concept_examples/k_means/gem_collector/model_of_interest_target_class_balanced_observations/batch_{batch}/layer_{layer_i}/k_{k}_cluster_labels.npy"
            )
            biases = []
            for cav_idx, cav in enumerate(cavs):
                mask_pos = cluster_labels == cav_idx
                mask_neg = cluster_labels != cav_idx

                observations_pos = observations[mask_pos]
                observations_neg = observations[mask_neg]

                if len(observations_neg) > len(observations_pos):
                    rng = np.random.default_rng(seed=28 + cav_idx)
                    indices = rng.choice(
                        len(observations_neg), size=len(observations_pos), replace=False
                    )
                    observations_neg = observations_neg[indices]

                pos_activations = mao.get_layer_activations(
                    layer_index=layer_i, model_inputs=observations_pos, flatten=True
                )
                neg_activations = mao.get_layer_activations(
                    layer_index=layer_i, model_inputs=observations_neg, flatten=True
                )

                assert (
                    pos_activations.shape[1] == cav.shape[0]
                ), "Mismatch in activation and CAV dimensions"

                pos_scores = pos_activations @ cav
                neg_scores = neg_activations @ cav

                bias = -0.5 * (np.mean(pos_scores) + np.mean(neg_scores))
                biases.append(bias)

            biases = np.array(biases)

            completeness_score, dt_model = ccm.train_and_eval_ccm(
                cavs=cavs,
                conv_handling="flatten",
                layer_i=layer_i,
                use_sigmoid=[True for _ in range(len(cavs))],  # All concepts are binary
                biases=biases,
            )

            results.append(
                {
                    "batch": batch,
                    "layer": layer_i,
                    "k": k,
                    "target_type": "all_q",
                    "max_depth": 3,
                    "completeness_score": completeness_score,
                }
            )

            if completeness_score > best_completeness_score:
                best_completeness_score = completeness_score
                best_dt_model = dt_model
                best_params = {"layer_i": layer_i, "k": k, "batch": batch}

            pbar.update(1)

    joblib.dump(
        best_dt_model,
        f"{save_path}best_ccm_k_means_layer_{best_params['layer_i']}_k_{best_params['k']}_batch_{best_params['batch']}_all_q_max_depth_3.joblib",
    )

    print(
        f"→ Best completeness={best_completeness_score:.4f} "
        f"(batch={best_params['batch']}, layer={best_params['layer_i']}, k={best_params['k']})"
    )

    df = pd.DataFrame(results)
    df.to_csv(f"{save_path}k_means_concepts_completeness_scores.csv", index=False)


if __name__ == "__main__":
    eval_k_means_concepts_ccm_dt()
