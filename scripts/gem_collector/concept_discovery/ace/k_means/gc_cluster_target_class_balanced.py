import numpy as np

from .gc_k_means_cluster import gc_k_means_cluster

np.random.seed(28)

if __name__ == "__main__":
    class_observations = np.load(
        "rl_concept_discovery_data/class_datasets_model_of_interest/gem_collector/target_class_balanced_30000_shuffled_examples.npy"
    )

    gc_k_means_cluster(
        class_observations=class_observations,
        save_directory_path="rl_ace_data/concept_examples/gem_collector/model_of_interest_target_class_balanced_observations/",
    )
