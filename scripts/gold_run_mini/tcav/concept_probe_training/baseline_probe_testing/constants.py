CONCEPT_PREFIXES = [
    "binary_concept_gold_above",
    "binary_concept_lava_1_above",
    "binary_concept_gold_above_and_lava_1_above",
    "binary_concept_gold_above_and_not_lava_1_above",
]
CONCEPT_NAMES = [
    "gold_above",
    "lava_1_above",
    "gold_above_and_lava_1_above",
    "gold_above_and_not_lava_1_above",
]
VALIDATION_SET_TARGET_SIZES = [400, 300, 15, 400]
CONCEPT_DATA_APPROACH_PATH_BASE = "rl_tcav_data/concept_examples/gold_run_mini/"
VALIDATION_DATASET_PATHS = [
    "rl_tcav_data/cavs/baseline_concept_probes_experiment/gold_run_mini/validation_sets/gold_above_probe_validation_dataset_method1.npy",
    "rl_tcav_data/cavs/baseline_concept_probes_experiment/gold_run_mini/validation_sets/lava_1_above_probe_validation_dataset_method2.npy",
    "rl_tcav_data/cavs/baseline_concept_probes_experiment/gold_run_mini/validation_sets/gold_above_and_lava_1_above_probe_validation_dataset_method1.npy",
    "rl_tcav_data/cavs/baseline_concept_probes_experiment/gold_run_mini/validation_sets/gold_above_and_not_lava_1_above_probe_validation_dataset_method1.npy",
]
VALIDATION_LABEL_SET_PATHS = [
    "rl_tcav_data/cavs/baseline_concept_probes_experiment/gold_run_mini/validation_sets/gold_above_probe_validation_label_set_method1.npy",
    "rl_tcav_data/cavs/baseline_concept_probes_experiment/gold_run_mini/validation_sets/lava_1_above_probe_validation_label_set_method2.npy",
    "rl_tcav_data/cavs/baseline_concept_probes_experiment/gold_run_mini/validation_sets/gold_above_and_lava_1_above_probe_validation_label_set_method1.npy",
    "rl_tcav_data/cavs/baseline_concept_probes_experiment/gold_run_mini/validation_sets/gold_above_and_not_lava_1_above_probe_validation_label_set_method1.npy",
]
