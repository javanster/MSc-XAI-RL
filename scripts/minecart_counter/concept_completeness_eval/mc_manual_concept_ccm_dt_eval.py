import os
import re

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


def eval_manual_concepts_ccm_dt():
    model: Sequential = load_model(MODEL_OF_INTEREST_PATH)  # type: ignore
    env = gym.make(id="MinecartCounter-v2", scatter_minecarts=True)
    mao = ModelActivationObtainer(model=model, input_normalization_type="image")

    X_train = np.load(
        "rl_ccm_data/obs_action_set/minecart_counter/kind-cosmos-35/X_train_8000_examples.npy"
    )
    X_val = np.load(
        "rl_ccm_data/obs_action_set/minecart_counter/kind-cosmos-35/X_val_2000_examples.npy"
    )

    save_path = f"{CCM_SCORES_DIR_PATH}/ccm_dt/"
    ensure_directory_exists(directory_path=save_path)

    layers = [0, 1, 3, 4, 5]
    results: list[dict] = []
    best_score = -np.inf
    best_model = None
    best_params: dict = {}
    best_cav_n = 0
    best_used_names: list[str] = []

    with tqdm(total=len(layers), unit="layer") as pbar:
        for layer_i in layers:
            cav_list: list[np.ndarray] = []
            bias_list: list[float] = []
            use_sigmoid: list[bool] = []
            used_names: list[str] = []

            probe_dir = (
                "rl_tcav_data/concept_probes/minecart_counter/"
                "completeness_testing/kind-cosmos-35/"
            )
            for fn in sorted(os.listdir(probe_dir)):
                if not fn.endswith(f"_layer_{layer_i}_concept_probe.keras"):
                    continue
                m = re.match(r"(.+)_layer_{}_concept_probe\.keras".format(layer_i), fn)
                if not m:
                    raise ValueError(f"Cannot parse concept name from {fn}")
                used_names.append(m.group(1))

                probe: Sequential = load_model(os.path.join(probe_dir, fn))  # type: ignore
                w, b = probe.layers[0].get_weights()
                cav_list.append(w.flatten())
                bias_list.append(b.flatten()[0])
                use_sigmoid.append(not fn.startswith("minecarts_n_"))

            cavs = np.stack(cav_list)
            biases = np.array(bias_list)
            cav_n = cavs.shape[0]

            Y_train = np.load(
                "rl_ccm_data/obs_action_set/minecart_counter/kind-cosmos-35/"
                "Y_all_q_train_8000_examples.npy"
            )
            Y_val = np.load(
                "rl_ccm_data/obs_action_set/minecart_counter/kind-cosmos-35/"
                "Y_all_q_val_2000_examples.npy"
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

            score, dt_model = ccm.train_and_eval_ccm(
                cavs=cavs,
                conv_handling="flatten",
                layer_i=layer_i,
                use_sigmoid=use_sigmoid,
                biases=biases,
            )

            results.append(
                {
                    "layer": layer_i,
                    "c": cav_n,
                    "target_type": "all_q",
                    "max_depth": 3,
                    "completeness_score": score,
                }
            )

            if score > best_score:
                best_score = score
                best_model = dt_model
                best_params = {"layer_i": layer_i}
                best_cav_n = cav_n
                best_used_names = used_names.copy()

            pbar.update(1)

    # save best DT
    joblib.dump(
        best_model,
        os.path.join(
            save_path,
            f"best_ccm_manual_layer_{best_params['layer_i']}"
            f"_c_{best_cav_n}_all_q_max_depth_3.joblib",
        ),
    )

    # save concept names
    pd.DataFrame(
        {"feature_index": list(range(len(best_used_names))), "concept_name": best_used_names}
    ).to_csv(os.path.join(save_path, "manual_concepts_used_names_best_layer.csv"), index=False)

    # save all scores
    pd.DataFrame(results).to_csv(
        os.path.join(save_path, "manual_concepts_completeness_scores.csv"), index=False
    )


if __name__ == "__main__":
    eval_manual_concepts_ccm_dt()
