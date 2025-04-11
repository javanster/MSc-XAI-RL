import glob

import keras
import numpy as np
import pandas as pd
from tqdm import tqdm

from rl_ace import ConceptProbeScoreKSelector, SilhouetteScoreKSelector
from utils import ModelActivationObtainer, ensure_directory_exists

from ...constants import MODEL_OF_INTEREST_PATH

TARGET_CLASS = "left"

if __name__ == "__main__":
    base_path = f"/Volumes/work/rl_tcav_data/concept_examples/gem_collector/ace_k_means/model_of_interest_target_class_{TARGET_CLASS}_observations/dim_reduction/"

    cps_k_selector = ConceptProbeScoreKSelector()
    ss_k_selector = SilhouetteScoreKSelector()

    model: keras.models.Sequential = keras.saving.load_model(MODEL_OF_INTEREST_PATH)  # type: ignore
    mao = ModelActivationObtainer(model=model, input_normalization_type="image")

    data = []

    total_iterations = (len(model.layers) - 2) * 49
    with tqdm(total=total_iterations, unit="Score calculation") as pbar:
        for layer_i in range(len(model.layers)):
            if layer_i == 2 or layer_i == 6:
                continue
            layer_path = f"{base_path}layer_{layer_i}/"
            cps_k_selector.reset()
            ss_k_selector.reset()
            for k in range(2, 51):
                k_path = f"{layer_path}k_{k}/"
                all_activations = []
                all_labels = []

                for cluster_i in range(k):
                    prefix = f"{k_path}layer_{layer_i}_cluster_{cluster_i}_"
                    pattern = prefix + "*.npy"
                    file_list = glob.glob(pattern)
                    if len(file_list) != 1:
                        raise ValueError(
                            f"Expected exactly one file matching {pattern}, but found {len(file_list)}"
                        )
                    file_name = file_list[0]
                    cluster_observations = np.load(file_name)
                    activations = mao.get_layer_activations(
                        layer_index=layer_i, flatten=True, model_inputs=cluster_observations
                    )

                    all_activations.extend(activations)
                    all_labels.extend([cluster_i for _ in range(len(activations))])

                cps_k_selector.assess_score_of_cluster(
                    activations=np.array(all_activations),
                    labels=np.array(all_labels),
                    k=k,
                    distances_from_centroids=np.array([]),
                )
                ss_k_selector.assess_score_of_cluster(
                    activations=np.array(all_activations),
                    labels=np.array(all_labels),
                    k=k,
                    distances_from_centroids=np.array([]),
                )
                pbar.update(1)

                data.append(
                    {
                        "layer": layer_i,
                        "k": k,
                        "weighted_average_concept_probe_score": cps_k_selector.score_values[-1],
                        "silhouette_score": ss_k_selector.score_values[-1],
                    }
                )

    data_df = pd.DataFrame(data)

    data_base_save_path = f"rl_ace_data/concept_examples/gem_collector/k_means/model_of_interest_target_class_{TARGET_CLASS}_observations/dim_reduction/"
    ensure_directory_exists(data_base_save_path)
    data_df.to_csv(f"{data_base_save_path}k_evaluation.csv", index=False)
