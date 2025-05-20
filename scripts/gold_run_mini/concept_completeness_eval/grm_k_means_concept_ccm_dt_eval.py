import itertools
import os

import gold_run_mini
import gymnasium as gym
import joblib
import numpy as np
import pandas as pd
from keras.api.models import Sequential
from keras.api.saving import load_model
from tqdm import tqdm

from rl_ccm import CCM_DT
from utils import ModelActivationObtainer, ensure_directory_exists

from .constants import CCM_SCORES_DIR_PATH, ENV_LAVA_SPOTS, MODEL_OF_INTEREST_PATH


def eval_k_means_concepts_ccm_dt():
    model: Sequential = load_model(MODEL_OF_INTEREST_PATH)  # type: ignore
    env = gym.make(id="GoldRunMini-v1", lava_spots=ENV_LAVA_SPOTS)
    mao = ModelActivationObtainer(model=model, input_normalization_type="image")

    save_path = f"{CCM_SCORES_DIR_PATH}ccm_dt/"
    ensure_directory_exists(directory_path=save_path)

    batches = list(range(100))
    layers = [0, 1, 3, 4, 5]
    k_values = list(range(2, 51))

    observations = np.load(
        "rl_concept_discovery_data/class_datasets_model_of_interest/"
        "gold_run_mini/target_class_balanced_6900_shuffled_examples.npy"
    )

    results = []
    best_score = -np.inf
    best_model = None
    best_params = {}

    total = len(batches) * len(layers) * len(k_values)
    pbar = tqdm(total=total, unit="score")

    for batch, layer_i, k in itertools.product(batches, layers, k_values):
        X_train = np.load(
            f"rl_ccm_data/obs_action_set/gold_run_mini/firm-mountain-13/"
            f"batch_{batch}/X_train_8000_examples.npy"
        )
        X_val = np.load(
            f"rl_ccm_data/obs_action_set/gold_run_mini/firm-mountain-13/"
            f"batch_{batch}/X_val_2000_examples.npy"
        )
        Y_train = np.load(
            f"rl_ccm_data/obs_action_set/gold_run_mini/firm-mountain-13/"
            f"batch_{batch}/Y_all_q_train_8000_examples.npy"
        )
        Y_val = np.load(
            f"rl_ccm_data/obs_action_set/gold_run_mini/firm-mountain-13/"
            f"batch_{batch}/Y_all_q_val_2000_examples.npy"
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

        base = (
            "/Volumes/work/rl_ace_data/concept_examples/k_means/"
            "gold_run_mini/model_of_interest_target_class_balanced_observations"
        )
        cavs = np.load(
            os.path.join(
                base,
                f"batch_{batch}",
                f"layer_{layer_i}",
                f"k_{k}_cluster_centroids.npy",
            )
        )
        cluster_labels = np.load(
            os.path.join(
                base,
                f"batch_{batch}",
                f"layer_{layer_i}",
                f"k_{k}_cluster_labels.npy",
            )
        )

        biases = []
        for cav_idx, cav in enumerate(cavs):
            mask_pos = cluster_labels == cav_idx
            obs_pos = observations[mask_pos]
            obs_neg = observations[~mask_pos]
            if len(obs_neg) > len(obs_pos):
                rng = np.random.default_rng(seed=28 + cav_idx)
                obs_neg = obs_neg[rng.choice(len(obs_neg), size=len(obs_pos), replace=False)]

            pos_act = mao.get_layer_activations(
                layer_index=layer_i, model_inputs=obs_pos, flatten=True
            )
            neg_act = mao.get_layer_activations(
                layer_index=layer_i, model_inputs=obs_neg, flatten=True
            )

            pos_scores = pos_act @ cav
            neg_scores = neg_act @ cav
            biases.append(-0.5 * (pos_scores.mean() + neg_scores.mean()))

        biases = np.array(biases)

        score, dt_model = ccm.train_and_eval_ccm(
            cavs=cavs,
            conv_handling="dim_reduction",
            layer_i=layer_i,
            use_sigmoid=[True] * len(cavs),
            biases=biases,
        )

        results.append(
            {
                "batch": batch,
                "layer": layer_i,
                "k": k,
                "target_type": "all_q",
                "max_depth": 3,
                "completeness_score": score,
            }
        )

        if score > best_score:
            best_score = score
            best_model = dt_model
            best_params = {"batch": batch, "layer_i": layer_i, "k": k}

        pbar.update(1)

    pbar.close()

    joblib.dump(
        best_model,
        os.path.join(
            save_path,
            f"best_ccm_k_means_batch_{best_params['batch']}"
            f"_layer_{best_params['layer_i']}_k_{best_params['k']}"
            f"_all_q_max_depth_3.joblib",
        ),
    )

    print(
        f"→ Best completeness={best_score:.4f} "
        f"(batch={best_params['batch']}, layer={best_params['layer_i']}, k={best_params['k']})"
    )

    pd.DataFrame(results).to_csv(
        os.path.join(save_path, "k_means_concepts_completeness_scores.csv"),
        index=False,
    )


if __name__ == "__main__":
    eval_k_means_concepts_ccm_dt()
