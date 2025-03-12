CONCEPT_PREFIXES = ["continuous_concept_minecarts_n", "binary_concept_minecart_1_left"]
CONCEPT_NAMES = ["minecarts_n", "minecart_1_left"]
VALIDATION_SET_TARGET_SIZES = [7_000, 600]
CONCEPT_DATA_APPROACH_PATH_BASE = "rl_tcav_data/concept_examples/minecart_counter/"
VALIDATION_DATASET_PATHS = [
    "rl_tcav_data/cavs/baseline_concept_probes_experiment/minecart_counter/validation_sets/minecarts_n_probe_validation_dataset_method1.npy",
    "rl_tcav_data/cavs/baseline_concept_probes_experiment/minecart_counter/validation_sets/minecart_1_left_probe_validation_dataset_method1.npy",
]
VALIDATION_LABEL_SET_PATHS = [
    "rl_tcav_data/cavs/baseline_concept_probes_experiment/minecart_counter/validation_sets/minecarts_n_probe_validation_label_set_method1.npy",
    "rl_tcav_data/cavs/baseline_concept_probes_experiment/minecart_counter/validation_sets/minecart_1_left_probe_validation_label_set_method1.npy",
]
