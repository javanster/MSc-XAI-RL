import itertools
import os

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


def eval_manual_concepts_ccm_dt(layer_i: int):
    model: Sequential = load_model(MODEL_OF_INTEREST_PATH)  # type: ignore
    env = gym.make(id="GemCollector-v3")
    mao = ModelActivationObtainer(model=model, input_normalization_type="image")

    X_train = np.load(
        "rl_ccm_data/obs_action_set/gem_collector/denim-sweep-56/X_train_8000_examples.npy"
    )
    X_val = np.load(
        "rl_ccm_data/obs_action_set/gem_collector/denim-sweep-56/X_val_2000_examples.npy"
    )

    results = []

    save_path = f"{CCM_SCORES_DIR_PATH}/ccm_dt/"

    ensure_directory_exists(directory_path=save_path)

    target_types = ["max_q", "all_q"]
    max_depths = [3, 4, 5]

    with tqdm(total=len(target_types) * len(max_depths), unit="score") as pbar:
        for target_type, max_depth in itertools.product(target_types, max_depths):

            Y_train = np.load(
                f"rl_ccm_data/obs_action_set/gem_collector/denim-sweep-56/Y_{target_type}_train_8000_examples.npy"
            )
            Y_val = np.load(
                f"rl_ccm_data/obs_action_set/gem_collector/denim-sweep-56/Y_{target_type}_val_2000_examples.npy"
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

            cav_n = 0

            cavs = []
            biases = []

            concept_probe_path = (
                "rl_tcav_data/concept_probes/gem_collector/completeness_testing/denim-sweep-56/"
            )
            for filename in sorted(os.listdir(concept_probe_path)):
                if filename.endswith(f"_layer_{layer_i}_concept_probe.keras"):
                    probe: Sequential = load_model(f"{concept_probe_path}{filename}")  # type: ignore
                    weights, bias = probe.layers[0].get_weights()
                    cav = weights.flatten()
                    cavs.append(cav)
                    biases.append(bias)

            cavs = np.array(cavs)
            biases = np.array([b.flatten() for b in biases])

            if cav_n != 0 and len(cavs) != cav_n:
                raise ValueError("Dissimilar number of cavs for all layers")
            cav_n = len(cavs)

            completeness_score, model = ccm.train_and_eval_ccm(
                cavs=cavs,
                conv_handling="flatten",
                layer_i=layer_i,
                output_probability=True,
                biases=biases,
            )

            results.append(
                {
                    "layer": layer_i,
                    "c": cav_n,
                    "target_type": target_type,
                    "max_depth": max_depth,
                    "completeness_score": completeness_score,
                }
            )

            joblib.dump(
                model,
                f"{save_path}best_ccm_manual_layer_{layer_i}_c_{cav_n}_{target_type}_max_depth_{max_depth}.joblib",
            )

            pbar.update(1)

    df = pd.DataFrame(results)
    df.to_csv(f"{save_path}manual_concepts_completeness_scores.csv", index=False)


if __name__ == "__main__":
    eval_manual_concepts_ccm_dt(layer_i=4)  # Based on best layer of nn ccm
