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

MISALIGNED = {
    "gem_collector": ("denim-sweep-56", 20),
    "minecart_counter": ("kind-cosmos-35", 15),
    "gold_run_mini": ("firm-mountain-13", 11),
}


def get_manual_cavs_and_names(
    manual_concept_probe_path: str,
    layer_i: int,
    environment_name: str,
    score_threshold: float = 0.5,
):
    model_name, _ = MISALIGNED[environment_name]
    scores_csv = os.path.join(
        "rl_tcav_data/concept_probes",
        environment_name,
        "completeness_testing",
        model_name,
        "concept_probe_scores.csv",
    )
    df = pd.read_csv(scores_csv)
    valid = set(
        df.query("layer_index == @layer_i and concept_probe_score >= @score_threshold")[
            "concept_name"
        ]
    )

    cavs, names = [], []
    for fn in sorted(os.listdir(manual_concept_probe_path)):
        if not fn.endswith(f"_layer_{layer_i}_concept_probe.keras"):
            continue
        name = fn[: fn.rfind(f"_layer_{layer_i}_concept_probe.keras")]
        if name not in valid:
            continue
        m: Sequential = load_model(os.path.join(manual_concept_probe_path, fn))  # type: ignore
        w, _ = m.layers[0].get_weights()
        cavs.append(w.flatten())
        names.append(name)

    if not cavs:
        raise RuntimeError(f"No manual CAVs found for {environment_name} layer {layer_i}")
    return np.stack(cavs), names


def eval_manual_concepts_ccm_dt():
    # load the trained model and environment
    model: Sequential = load_model(MODEL_OF_INTEREST_PATH)  # type: ignore
    env = gym.make(
        id="GoldRunMini-v1",
        lava_spots=ENV_LAVA_SPOTS,
    )
    mao = ModelActivationObtainer(model=model, input_normalization_type="image")

    # paths and data
    model_name, _ = MISALIGNED["gold_run_mini"]
    base_obs_path = f"rl_ccm_data/obs_action_set/gold_run_mini/{model_name}"
    X_train = np.load(os.path.join(base_obs_path, "X_train_8000_examples.npy"))
    X_val = np.load(os.path.join(base_obs_path, "X_val_2000_examples.npy"))

    save_path = os.path.join(CCM_SCORES_DIR_PATH, "ccm_dt") + os.sep
    ensure_directory_exists(directory_path=save_path)

    layers = [0, 1, 3, 4, 5]

    # track best models across layers
    best_all_score = -np.inf
    best_all_model = None
    best_all_params = {}
    best_all_names: list[str] = []

    best_filt_score = -np.inf
    best_filt_model = None
    best_filt_params = {}
    best_filt_names: list[str] = []

    results_all = []
    results_filtered = []

    concept_probe_base = os.path.join(
        "rl_tcav_data", "concept_probes", "gold_run_mini", "completeness_testing", model_name
    )

    with tqdm(total=len(layers), unit="layer") as pbar:
        for layer_i in layers:
            # --- ALL manual concepts ---
            all_cavs = []
            all_biases = []
            all_names = []
            for fn in sorted(os.listdir(concept_probe_base)):
                if not fn.endswith(f"_layer_{layer_i}_concept_probe.keras"):
                    continue
                name = fn[: fn.rfind(f"_layer_{layer_i}_concept_probe.keras")]
                probe: Sequential = load_model(os.path.join(concept_probe_base, fn))  # type: ignore
                w, b = probe.layers[0].get_weights()
                all_cavs.append(w.flatten())
                all_biases.append(b.flatten())
                all_names.append(name)
            all_cavs = np.stack(all_cavs)
            all_biases = np.stack(all_biases)
            c_all = all_cavs.shape[0]

            ccm_all = CCM_DT(
                num_classes=env.action_space.n,
                model_activation_obtainer=mao,
                X_train=X_train,
                X_val=X_val,
                Y_train=np.load(os.path.join(base_obs_path, "Y_all_q_train_8000_examples.npy")),
                Y_val=np.load(os.path.join(base_obs_path, "Y_all_q_val_2000_examples.npy")),
                all_q=True,
                max_depth=3,
            )
            score_all, dt_all = ccm_all.train_and_eval_ccm(
                cavs=all_cavs,
                conv_handling="flatten",
                layer_i=layer_i,
                use_sigmoid=[True] * c_all,
                biases=all_biases,
            )
            results_all.append(
                {
                    "layer": layer_i,
                    "c": c_all,
                    "target_type": "all_q",
                    "max_depth": 3,
                    "completeness_score": score_all,
                }
            )
            if score_all > best_all_score:
                best_all_score = score_all
                best_all_model = dt_all
                best_all_params = {"layer": layer_i, "c": c_all}
                best_all_names = all_names.copy()

            # --- FILTERED manual concepts ---
            cavs_filt, names_filt = get_manual_cavs_and_names(
                manual_concept_probe_path=concept_probe_base,
                layer_i=layer_i,
                environment_name="gold_run_mini",
                score_threshold=0.5,
            )
            biases_filt = []
            for name in names_filt:
                fn = f"{name}_layer_{layer_i}_concept_probe.keras"
                probe = load_model(os.path.join(concept_probe_base, fn))  # type: ignore
                _, b = probe.layers[0].get_weights()
                biases_filt.append(b.flatten())
            biases_filt = np.stack(biases_filt)
            c_filt = cavs_filt.shape[0]

            ccm_filt = CCM_DT(
                num_classes=env.action_space.n,
                model_activation_obtainer=mao,
                X_train=X_train,
                X_val=X_val,
                Y_train=np.load(os.path.join(base_obs_path, "Y_all_q_train_8000_examples.npy")),
                Y_val=np.load(os.path.join(base_obs_path, "Y_all_q_val_2000_examples.npy")),
                all_q=True,
                max_depth=3,
            )
            score_filt, dt_filt = ccm_filt.train_and_eval_ccm(
                cavs=cavs_filt,
                conv_handling="flatten",
                layer_i=layer_i,
                use_sigmoid=[True] * c_filt,
                biases=biases_filt,
            )
            results_filtered.append(
                {
                    "layer": layer_i,
                    "c": c_filt,
                    "target_type": "all_q",
                    "max_depth": 3,
                    "completeness_score": score_filt,
                }
            )
            if score_filt > best_filt_score:
                best_filt_score = score_filt
                best_filt_model = dt_filt
                best_filt_params = {"layer": layer_i, "c": c_filt}
                best_filt_names = names_filt.copy()

            pbar.update(1)

    # save best models & names
    joblib.dump(
        best_all_model,
        os.path.join(
            save_path,
            f"best_manual_all_layer_{best_all_params['layer']}_c_{best_all_params['c']}_all_q_max_depth_3.joblib",
        ),
    )
    pd.DataFrame(
        {
            "feature_index": list(range(len(best_all_names))),
            "concept_name": best_all_names,
        }
    ).to_csv(
        os.path.join(save_path, "manual_concepts_used_names_best_all.csv"),
        index=False,
    )

    joblib.dump(
        best_filt_model,
        os.path.join(
            save_path,
            f"best_manual_filtered_layer_{best_filt_params['layer']}_c_{best_filt_params['c']}_all_q_max_depth_3.joblib",
        ),
    )
    pd.DataFrame(
        {
            "feature_index": list(range(len(best_filt_names))),
            "concept_name": best_filt_names,
        }
    ).to_csv(
        os.path.join(save_path, "manual_concepts_used_names_best_filtered.csv"),
        index=False,
    )

    # save results
    pd.DataFrame(results_all).to_csv(
        os.path.join(save_path, "manual_concepts_completeness_scores_all.csv"),
        index=False,
    )
    pd.DataFrame(results_filtered).to_csv(
        os.path.join(save_path, "manual_concepts_completeness_scores_filtered.csv"),
        index=False,
    )


if __name__ == "__main__":
    eval_manual_concepts_ccm_dt()
