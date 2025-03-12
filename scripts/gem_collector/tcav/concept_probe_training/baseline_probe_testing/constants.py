CONCEPT_NAMES = ["aquamarine_left", "lava_1_above"]
CONCEPT_PREFIXES = ["binary_concept_aquamarine_left", "binary_concept_lava_1_above"]
VALIDATION_SET_TARGET_SIZES = [16_000, 24]
CONCEPT_DATA_APPROACH_PATH_BASE = "rl_tcav_data/concept_examples/gem_collector/"
VALIDATION_DATASET_PATHS = [
    "rl_tcav_data/cavs/baseline_concept_probes_experiment/gem_collector/validation_sets/aquamarine_left_probe_validation_dataset_method1.npy",
    "rl_tcav_data/cavs/baseline_concept_probes_experiment/gem_collector/validation_sets/lava_1_above_probe_validation_dataset_method2.npy",
]
VALIDATION_LABEL_SET_PATHS = [
    "rl_tcav_data/cavs/baseline_concept_probes_experiment/gem_collector/validation_sets/aquamarine_left_probe_validation_label_set_method1.npy",
    "rl_tcav_data/cavs/baseline_concept_probes_experiment/gem_collector/validation_sets/lava_1_above_probe_validation_label_set_method2.npy",
]
