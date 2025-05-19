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


def eval_ice_concepts_ccm_dt():
    model: Sequential = load_model(MODEL_OF_INTEREST_PATH)  # type: ignore
    env = gym.make(id="GemCollector-v3")
    mao = ModelActivationObtainer(model=model, input_normalization_type="image")

    save_path = f"{CCM_SCORES_DIR_PATH}/ccm_dt/"
    ensure_directory_exists(directory_path=save_path)

    batches = list(range(100))
    layers = [1]
    target_type = "all_q"
    max_depth = 3

    cs_per_batch_layer = {}
    total = 0
    base_dir = "/Volumes/work/rl_ice_data/ncavs/gem_collector/model_of_interest_target_class_balanced_observations"
    for batch in batches:
        for layer_i in layers:
            layer_dir = os.path.join(base_dir, f"batch_{batch}", f"layer_{layer_i}")
            cs = sorted(
                int(m.group(1))
                for fn in os.listdir(layer_dir)
                if (m := re.match(r"c_(\d+)_ncavs\.npy", fn))
            )
            cs_per_batch_layer[(batch, layer_i)] = cs
            total += len(cs)

    results = []
    best_score = -np.inf
    best_model = None
    best_params = {}

    pbar = tqdm(total=total, unit="run")
    for batch in batches:
        X_train = np.load(
            f"/Volumes/work/rl_ccm_data/obs_action_set/gem_collector/denim-sweep-56/"
            f"batch_{batch}/X_train_8000_examples.npy"
        )
        X_val = np.load(
            f"/Volumes/work/rl_ccm_data/obs_action_set/gem_collector/denim-sweep-56/"
            f"batch_{batch}/X_val_2000_examples.npy"
        )
        Y_train = np.load(
            f"/Volumes/work/rl_ccm_data/obs_action_set/gem_collector/denim-sweep-56/"
            f"batch_{batch}/Y_{target_type}_train_8000_examples.npy"
        )
        Y_val = np.load(
            f"/Volumes/work/rl_ccm_data/obs_action_set/gem_collector/denim-sweep-56/"
            f"batch_{batch}/Y_{target_type}_val_2000_examples.npy"
        )

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
            cs = cs_per_batch_layer[(batch, layer_i)]
            for c in cs:
                pbar.set_description(f"batch={batch} layer={layer_i} c={c}")
                cavs = np.load(
                    os.path.join(base_dir, f"batch_{batch}", f"layer_{layer_i}", f"c_{c}_ncavs.npy")
                )

                completeness_score, dt_model = ccm.train_and_eval_ccm(
                    cavs=cavs,
                    conv_handling="dim_reduction",
                    layer_i=layer_i,
                    use_sigmoid=[False] * len(cavs),
                    biases=None,
                )

                results.append(
                    {
                        "batch": batch,
                        "layer": layer_i,
                        "c": c,
                        "target_type": target_type,
                        "max_depth": max_depth,
                        "completeness_score": completeness_score,
                    }
                )

                if completeness_score > best_score:
                    best_score = completeness_score
                    best_model = dt_model
                    best_params = {"batch": batch, "layer": layer_i, "c": c}

                pbar.update(1)

    pbar.close()

    joblib.dump(
        best_model,
        os.path.join(
            save_path,
            f"best_ccm_ice_batch_{best_params['batch']}"
            f"_layer_{best_params['layer']}_c_{best_params['c']}"
            f"_{target_type}_max_depth_{max_depth}.joblib",
        ),
    )

    pd.DataFrame(results).to_csv(
        os.path.join(save_path, "ice_concepts_completeness_scores.csv"), index=False
    )


if __name__ == "__main__":
    eval_ice_concepts_ccm_dt()
