import numpy as np
from keras.api.saving import load_model

from rl_ace import FCMeansClusterer
from utils import ModelActivationObtainer

from ..constants import MAX_CLUSTERS, MODEL_LAYERS_OF_INTEREST, MODEL_OF_INTEREST_PATH


def fc_means_cluster(class_observations: np.ndarray, save_directory_path: str) -> None:
    moi: Sequential = load_model(MODEL_OF_INTEREST_PATH)  # type: ignore
    mao = ModelActivationObtainer(model=moi, input_normalization_type="image")

    fc_means_example_clusterer = FCMeansClusterer(model_activation_obtainer=mao)

    fc_means_example_clusterer.cluster(
        environment_observations=class_observations,
        max_c=MAX_CLUSTERS,
        model_layer_indexes=MODEL_LAYERS_OF_INTEREST,
        save_directory_path=save_directory_path,
    )


if __name__ == "__main__":
    target_classes = [
        "balanced",
        "left",
        "right",
        "do_nothing",
    ]

    for target_class in target_classes:
        class_observations = np.load(
            f"rl_concept_discovery_data/class_datasets_model_of_interest/gem_collector/target_class_{target_class}_30000_shuffled_examples.npy"
        )

        fc_means_cluster(
            class_observations=class_observations,
            save_directory_path=f"rl_ace_data/concept_examples/fuzzy_c_means/gem_collector/model_of_interest_target_class_{target_class}_observations/",
        )
