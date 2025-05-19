import os
import re

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


def eval_manual_concepts_ccm_dt():
    model: Sequential = load_model(MODEL_OF_INTEREST_PATH)  # type: ignore
    env = gym.make(id="GemCollector-v3")
    mao = ModelActivationObtainer(model=model, input_normalization_type="image")

    X_train = np.load(
        "rl_ccm_data/obs_action_set/gem_collector/denim-sweep-56/X_train_8000_examples.npy"
    )
    X_val = np.load(
        "rl_ccm_data/obs_action_set/gem_collector/denim-sweep-56/X_val_2000_examples.npy"
    )

    save_path = f"{CCM_SCORES_DIR_PATH}ccm_dt/"
    ensure_directory_exists(directory_path=save_path)

    layers = [0, 1, 3, 4, 5]

    results = []
    best_completeness_score = -np.inf
    best_dt_model = None
    best_params = {}
    # Initialize these so they're always defined
    best_cav_n = 0
    best_used_concept_names: list[str] = []

    with tqdm(total=len(layers), unit="layer") as pbar:
        for layer_i in layers:

            cavs = []
            biases = []
            used_concept_names: list[str] = []

            concept_probe_path = (
                "rl_tcav_data/concept_probes/gem_collector/completeness_testing/denim-sweep-56/"
            )
            for filename in sorted(os.listdir(concept_probe_path)):
                if filename.endswith(f"_layer_{layer_i}_concept_probe.keras"):
                    m = re.match(r"(.+)_layer_{}_concept_probe\.keras".format(layer_i), filename)
                    if not m:
                        raise ValueError(f"Cannot parse concept name from {filename}")
                    concept_name = m.group(1)
                    used_concept_names.append(concept_name)

                    probe: Sequential = load_model(os.path.join(concept_probe_path, filename))  # type: ignore
                    weights, bias = probe.layers[0].get_weights()
                    cavs.append(weights.flatten())
                    biases.append(bias.flatten())

            cavs = np.array(cavs)
            biases = np.array(biases)
            cav_n = len(cavs)

            Y_train = np.load(
                "rl_ccm_data/obs_action_set/gem_collector/denim-sweep-56/Y_all_q_train_8000_examples.npy"
            )
            Y_val = np.load(
                "rl_ccm_data/obs_action_set/gem_collector/denim-sweep-56/Y_all_q_val_2000_examples.npy"
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

            completeness_score, dt_model = ccm.train_and_eval_ccm(
                cavs=cavs,
                conv_handling="flatten",
                layer_i=layer_i,
                use_sigmoid=[True] * cav_n,  # all concepts binary
                biases=biases,
            )

            results.append(
                {
                    "layer": layer_i,
                    "c": cav_n,
                    "target_type": "all_q",
                    "max_depth": 3,
                    "completeness_score": completeness_score,
                }
            )

            # If this layer is the new best, stash its info
            if completeness_score > best_completeness_score:
                best_completeness_score = completeness_score
                best_dt_model = dt_model
                best_params = {"layer_i": layer_i}
                best_cav_n = cav_n
                best_used_concept_names = used_concept_names.copy()

            pbar.update(1)

    # Save the best decision-tree model
    joblib.dump(
        best_dt_model,
        os.path.join(
            save_path,
            f"best_ccm_manual_layer_{best_params['layer_i']}"
            f"_c_{best_cav_n}_all_q_max_depth_3.joblib",
        ),
    )

    concept_names_df = pd.DataFrame(
        {
            "feature_index": list(range(len(best_used_concept_names))),
            "concept_name": best_used_concept_names,
        }
    )
    concept_names_df.to_csv(
        os.path.join(save_path, "manual_concepts_used_names_best_layer.csv"), index=False
    )

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_path, "manual_concepts_completeness_scores.csv"), index=False)


if __name__ == "__main__":
    eval_manual_concepts_ccm_dt()
