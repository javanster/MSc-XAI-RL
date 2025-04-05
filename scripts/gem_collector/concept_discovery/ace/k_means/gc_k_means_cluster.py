import numpy as np
from keras.api.saving import load_model

from rl_ace import KMeansClusterer
from utils import ModelActivationObtainer

from .constants import MAX_K, MODEL_LAYERS_OF_INTEREST, MODEL_OF_INTEREST_PATH


def gc_k_means_cluster(class_observations: np.ndarray, save_directory_path: str) -> None:
    moi: Sequential = load_model(MODEL_OF_INTEREST_PATH)  # type: ignore
    mao = ModelActivationObtainer(model=moi, input_normalization_type="image")

    k_means_example_clusterer = KMeansClusterer(model_activation_obtainer=mao)

    k_means_example_clusterer.cluster(
        environment_observations=class_observations,
        max_k=MAX_K,
        model_layer_indexes=MODEL_LAYERS_OF_INTEREST,
        save_directory_path=save_directory_path,
    )
