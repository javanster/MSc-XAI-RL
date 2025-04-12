import os
import re
from typing import cast

import gem_collector
import gymnasium as gym
import numpy as np
import pandas as pd
from keras.api.models import Sequential
from keras.api.saving import load_model
from tqdm import tqdm

from rl_ccm import CCM_NN
from utils import ModelActivationObtainer, ensure_directory_exists

from .constants import CCM_SCORES_DIR_PATH, MODEL_LAYERS_OF_INTEREST, MODEL_OF_INTEREST_PATH


def eval_manual_concepts_ccm_nn():
    model: Sequential = load_model(MODEL_OF_INTEREST_PATH)  # type: ignore
    env = gym.make(id="GemCollector-v3")
    mao = ModelActivationObtainer(model=model, input_normalization_type="image")

    X_train = np.load(
        "rl_ccm_data/obs_action_set/gem_collector/denim-sweep-56/X_train_8000_examples.npy"
    )
    X_val = np.load(
        "rl_ccm_data/obs_action_set/gem_collector/denim-sweep-56/X_val_2000_examples.npy"
    )
    Y_train = np.load(
        "rl_ccm_data/obs_action_set/gem_collector/denim-sweep-56/Y_all_q_train_8000_examples.npy"
    )
    Y_val = np.load(
        "rl_ccm_data/obs_action_set/gem_collector/denim-sweep-56/Y_all_q_val_2000_examples.npy"
    )

    ccm = CCM_NN(
        num_classes=env.action_space.n,
        model_activation_obtainer=mao,
        X_train=X_train,
        X_val=X_val,
        Y_train=Y_train,
        Y_val=Y_val,
        all_q=True,
    )

    results = []
    completeness_scores = []
    ccm_models = []

    cav_ns = {}

    with tqdm(unit="score", total=len(MODEL_LAYERS_OF_INTEREST)) as pbar:
        for layer_i in MODEL_LAYERS_OF_INTEREST:

            cavs = []

            concept_probe_path = (
                "rl_tcav_data/concept_probes/gem_collector/completeness_testing/denim-sweep-56/"
            )
            for filename in sorted(os.listdir(concept_probe_path)):
                if filename.endswith(f"_layer_{layer_i}_concept_probe.keras"):
                    probe: Sequential = load_model(f"{concept_probe_path}{filename}")  # type: ignore
                    weights, _ = probe.layers[0].get_weights()
                    cav = weights.flatten()
                    cavs.append(cav)

            cavs = np.array(cavs)

            cav_ns[layer_i] = len(cavs)

            completeness_score, nn = ccm.train_and_eval_ccm(
                cavs=cavs,
                conv_handling="flatten",
                layer_i=layer_i,
            )

            nn = cast(Sequential, nn)

            completeness_scores.append(completeness_score)
            ccm_models.append(nn)

            results.append(
                {"layer": layer_i, "c": len(cavs), "completeness_score": completeness_score}
            )
            pbar.update(1)

    bcsi = np.argmax(completeness_scores)
    best_model = ccm_models[bcsi]
    best_result = results[bcsi]
    best_layer = best_result["layer"]
    cav_n = cav_ns[best_layer]

    save_path = f"{CCM_SCORES_DIR_PATH}/ccm_nn/"

    ensure_directory_exists(directory_path=save_path)

    best_model.save(
        f"{save_path}best_ccm_manual_layer_{best_layer}_c_{cav_n}.keras",
    )

    df = pd.DataFrame(results)
    df.to_csv(f"{save_path}manual_completeness_scores_all_q.csv", index=False)


if __name__ == "__main__":
    eval_manual_concepts_ccm_nn()
