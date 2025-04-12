import itertools
from typing import cast

import gem_collector
import gymnasium as gym
import joblib
import numpy as np
import pandas as pd
from keras.api.models import Sequential
from keras.api.saving import load_model
from tqdm import tqdm

from rl_ccm import CCM_NN
from utils import ModelActivationObtainer, ensure_directory_exists

from .constants import CCM_SCORES_DIR_PATH, MODEL_LAYERS_OF_INTEREST, MODEL_OF_INTEREST_PATH


def eval_ace_fc_means_concepts_nn():
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

    c_values = [c for c in range(2, 51)]
    results = []
    completeness_scores = []
    ccm_nns = []

    with tqdm(unit="score", total=len(MODEL_LAYERS_OF_INTEREST) * len(c_values)) as pbar:
        for layer_i, c in itertools.product(MODEL_LAYERS_OF_INTEREST, c_values):

            cavs = np.load(
                f"rl_ace_data/concept_examples/fuzzy_c_means/gem_collector/model_of_interest_target_class_balanced_observations/layer_{layer_i}/c_{c}_cluster_centroids.npy"
            )

            completeness_score, nn = ccm.train_and_eval_ccm(
                cavs=cavs,
                conv_handling="dim_reduction",
                layer_i=layer_i,
            )

            completeness_scores.append(completeness_score)

            nn = cast(Sequential, nn)
            ccm_nns.append(nn)

            results.append({"layer": layer_i, "c": c, "completeness_score": completeness_score})
            pbar.update(1)

    bcsi = np.argmax(completeness_scores)
    best_nn = ccm_nns[bcsi]
    best_result = results[bcsi]
    best_c = best_result["c"]
    best_layer = best_result["layer"]

    save_path = f"{CCM_SCORES_DIR_PATH}/ccm_nn/"

    ensure_directory_exists(directory_path=save_path)

    best_nn.save(
        f"{save_path}best_ccm_fc_means_layer_{best_layer}_c_{best_c}.keras",
    )

    df = pd.DataFrame(results)
    df.to_csv(f"{save_path}fc_means_completeness_scores_all_q.csv", index=False)


if __name__ == "__main__":
    eval_ace_fc_means_concepts_nn()
