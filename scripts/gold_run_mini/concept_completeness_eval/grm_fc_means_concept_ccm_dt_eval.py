import os
import re

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


def eval_fc_means_concepts_ccm_dt():
    model: Sequential = load_model(MODEL_OF_INTEREST_PATH)  # type: ignore
    env = gym.make(id="GoldRunMini-v1", lava_spots=ENV_LAVA_SPOTS)
    mao = ModelActivationObtainer(model=model, input_normalization_type="image")

    save_path = f"{CCM_SCORES_DIR_PATH}ccm_dt/"
    ensure_directory_exists(directory_path=save_path)

    batches = list(range(100))
    layers = [0, 1, 3, 4, 5]
    target_type = "all_q"
    max_depth = 3

    X_TMPL = "rl_ccm_data/obs_action_set/gold_run_mini/firm-mountain-13/" "batch_{batch}/{file}"
    FC_BASE = (
        "/Volumes/work/rl_ace_data/concept_examples/fuzzy_c_means/"
        "gold_run_mini/model_of_interest_target_class_balanced_observations"
    )
    observations = np.load(
        "rl_concept_discovery_data/class_datasets_model_of_interest/"
        "gold_run_mini/target_class_balanced_6900_shuffled_examples.npy"
    )

    # discover available c values per (batch, layer)
    cs_per = {}
    total = 0
    for batch in batches:
        for layer_i in layers:
            layer_dir = os.path.join(FC_BASE, f"batch_{batch}", f"layer_{layer_i}")
            cs = sorted(
                int(m.group(1))
                for fn in os.listdir(layer_dir)
                if (m := re.match(r"c_(\d+)_cluster_centroids\.npy", fn))
            )
            cs_per[(batch, layer_i)] = cs
            total += len(cs)

    results = []
    best_score = -np.inf
    best_model = None
    best_params = {}

    pbar = tqdm(total=total, unit="run")
    for batch in batches:
        # load RL data for this batch
        X_train = np.load(X_TMPL.format(batch=batch, file="X_train_8000_examples.npy"))
        X_val = np.load(X_TMPL.format(batch=batch, file="X_val_2000_examples.npy"))
        Y_train = np.load(
            X_TMPL.format(batch=batch, file=f"Y_{target_type}_train_8000_examples.npy")
        )
        Y_val = np.load(X_TMPL.format(batch=batch, file=f"Y_{target_type}_val_2000_examples.npy"))

        ccm = CCM_DT(
            num_classes=env.action_space.n,
            model_activation_obtainer=mao,
            X_train=X_train,
            X_val=X_val,
            Y_train=Y_train,
            Y_val=Y_val,
            all_q=True,
            max_depth=max_depth,
        )

        for layer_i in layers:
            for c in cs_per[(batch, layer_i)]:
                pbar.set_description(f"batch={batch} layer={layer_i} c={c}")

                # load centroids and hard labels
                centroids_path = os.path.join(
                    FC_BASE, f"batch_{batch}", f"layer_{layer_i}", f"c_{c}_cluster_centroids.npy"
                )
                labels_path = os.path.join(
                    FC_BASE, f"batch_{batch}", f"layer_{layer_i}", f"c_{c}_hard_cluster_labels.npy"
                )
                cavs = np.load(centroids_path)
                cluster_labels = np.load(labels_path)

                biases = []
                for cav_idx, cav in enumerate(cavs):
                    mask_pos = cluster_labels == cav_idx
                    obs_pos = observations[mask_pos]
                    # fallback: use centroid if no hard positives
                    if obs_pos.shape[0] == 0:
                        pos_act = cav[None, :]
                    else:
                        pos_act = mao.get_layer_activations(
                            layer_index=layer_i, model_inputs=obs_pos, flatten=True
                        )
                    n_pos = pos_act.shape[0]

                    # down-sample negatives to n_pos
                    obs_neg = observations[~mask_pos]
                    if len(obs_neg) > n_pos:
                        rng = np.random.default_rng(seed=28 + cav_idx)
                        idx = rng.choice(len(obs_neg), size=n_pos, replace=False)
                        obs_neg = obs_neg[idx]

                    neg_act = mao.get_layer_activations(
                        layer_index=layer_i, model_inputs=obs_neg, flatten=True
                    )

                    pos_scores = pos_act @ cav
                    neg_scores = neg_act @ cav
                    biases.append(-0.5 * (pos_scores.mean() + neg_scores.mean()))

                biases = np.array(biases)

                score, dt_model = ccm.train_and_eval_ccm(
                    cavs=cavs,
                    conv_handling="flatten",
                    layer_i=layer_i,
                    use_sigmoid=[True] * len(cavs),
                    biases=biases,
                )

                results.append(
                    {
                        "batch": batch,
                        "layer": layer_i,
                        "c": c,
                        "target_type": target_type,
                        "max_depth": max_depth,
                        "completeness_score": score,
                    }
                )

                if score > best_score:
                    best_score = score
                    best_model = dt_model
                    best_params = {"batch": batch, "layer": layer_i, "c": c}

                pbar.update(1)

    pbar.close()

    # save best model and summary
    joblib.dump(
        best_model,
        os.path.join(
            save_path,
            f"best_ccm_fc_means_batch_{best_params['batch']}"
            f"_layer_{best_params['layer']}_c_{best_params['c']}"
            f"_{target_type}_max_depth_{max_depth}.joblib",
        ),
    )

    pd.DataFrame(results).to_csv(
        os.path.join(save_path, "fc_means_concepts_completeness_scores.csv"), index=False
    )


if __name__ == "__main__":
    eval_fc_means_concepts_ccm_dt()
