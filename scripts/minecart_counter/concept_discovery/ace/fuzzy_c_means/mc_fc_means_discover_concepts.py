import numpy as np
from keras.api.saving import load_model

from rl_ace import FCMeansClusterer
from utils import ModelActivationObtainer

from ..constants import MAX_CLUSTERS, MODEL_LAYERS_OF_INTEREST, MODEL_OF_INTEREST_PATH


def mc_fc_means_cluster() -> None:
    moi: Sequential = load_model(MODEL_OF_INTEREST_PATH)  # type: ignore
    mao = ModelActivationObtainer(model=moi, input_normalization_type="image")

    target_classes = [
        "balanced",
        "up",
        "upright",
        "right",
        "downright",
        "down",
        "downleft",
        "left",
        "upleft",
    ]
    class_obs_paths = [
        "rl_concept_discovery_data/class_datasets_model_of_interest/minecart_counter/target_class_balanced_30000_shuffled_examples.npy",
        "rl_concept_discovery_data/class_datasets_model_of_interest/minecart_counter/target_class_up_7070_shuffled_examples.npy",
        "rl_concept_discovery_data/class_datasets_model_of_interest/minecart_counter/target_class_upright_30000_shuffled_examples.npy",
        "rl_concept_discovery_data/class_datasets_model_of_interest/minecart_counter/target_class_right_22997_shuffled_examples.npy",
        "rl_concept_discovery_data/class_datasets_model_of_interest/minecart_counter/target_class_downright_11686_shuffled_examples.npy",
        "rl_concept_discovery_data/class_datasets_model_of_interest/minecart_counter/target_class_down_9680_shuffled_examples.npy",
        "rl_concept_discovery_data/class_datasets_model_of_interest/minecart_counter/target_class_downleft_7738_shuffled_examples.npy",
        "rl_concept_discovery_data/class_datasets_model_of_interest/minecart_counter/target_class_left_12383_shuffled_examples.npy",
        "rl_concept_discovery_data/class_datasets_model_of_interest/minecart_counter/target_class_upleft_24696_shuffled_examples.npy",
    ]

    fc_means_example_clusterer = FCMeansClusterer(model_activation_obtainer=mao)

    for target_class, class_obs_path in zip(target_classes, class_obs_paths):
        class_observations = np.load(class_obs_path)

        fc_means_example_clusterer.cluster(
            environment_observations=class_observations,
            max_c=MAX_CLUSTERS,
            model_layer_indexes=MODEL_LAYERS_OF_INTEREST,
            save_directory_path=f"rl_ace_data/concept_examples/fuzzy_c_means/minecart_counter/model_of_interest_target_class_{target_class}_observations/",
        )


if __name__ == "__main__":
    mc_fc_means_cluster()
